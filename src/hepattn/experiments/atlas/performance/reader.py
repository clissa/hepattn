import uproot
import numpy as np
import awkward as ak
from tqdm import tqdm
from copy import deepcopy
import gc
from ..utility.helper_dicts import pdgid_class_dict, class_mass_dict


def load_predictions(pred_path, threshold=0.5, load_hung_matched_truth=False, model_name='hgpflow',
                     num_workers=32, entry_start=0, entry_stop=None, step_size=50_000):
    """Load HGPflow predictions from a single ROOT file.

    entry_start / entry_stop select a subset of events (default: whole tree).
    step_size sets the chunk size used for the threaded read + progress bar.
    """
    import concurrent.futures
    import time
    _t0 = time.perf_counter()

    tree = uproot.open(pred_path)['event_tree']

    vars_to_load = [
        'pred_ind', 'proxy_pt', 'proxy_eta', 'proxy_phi',
        f'{model_name}_pt', f'{model_name}_eta', f'{model_name}_phi', f'{model_name}_class']
    if model_name == 'hgpflow':
        vars_to_load += ['assoc_track_pt', 'assoc_track_eta', 'assoc_track_phi']
    elif model_name == 'mpflow':
        vars_to_load += ['proxy_is_charged',
                         'proxy_ch_pt',   'proxy_ch_eta',   'proxy_ch_phi',
                         'proxy_neut_pt', 'proxy_neut_eta', 'proxy_neut_phi']

    rename_dict = {}
    if load_hung_matched_truth:
        vars_to_load += ['truth_pt', 'truth_eta', 'truth_phi', 'truth_class']
        rename_dict = {
            'truth_pt': 'hung_matched_truth_pt', 'truth_eta': 'hung_matched_truth_eta',
            'truth_phi': 'hung_matched_truth_phi', 'truth_class': 'hung_matched_truth_class'}

    # Read the tree in chunks with explicit thread executors so uproot decompresses
    # and interprets baskets in parallel (was serial per-branch before). Iterating
    # instead of a single arrays() call gives us a progress bar and caps peak memory
    # to one chunk during the read.
    n_total = tree.num_entries
    if entry_stop is None:
        entry_stop = n_total
    entry_stop = min(entry_stop, n_total)
    n_read = max(0, entry_stop - entry_start)
    n_steps = max(1, -(-n_read // step_size))  # ceil division

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)
    print(f"\033[96mLoading {model_name} predictions: {n_read}/{n_total} events, "
          f"{num_workers} workers, step_size={step_size}\033[0m")
    chunks = list(tqdm(
        tree.iterate(
            vars_to_load + ['event_number'],
            library='ak',
            entry_start=entry_start,
            entry_stop=entry_stop,
            step_size=step_size,
            decompression_executor=executor,
            interpretation_executor=executor,
        ),
        total=n_steps, desc=f"Loading {model_name} predictions...",
    ))
    arr = ak.concatenate(chunks) if len(chunks) > 1 else chunks[0]
    print(f"\033[96m  read done in {time.perf_counter() - _t0:.1f}s\033[0m")

    pred_dict = {}
    mask = arr['pred_ind'] > threshold
    for var in vars_to_load:
        new_var = rename_dict.get(var, var)
        pred_dict[new_var] = ak.drop_none(ak.mask(arr[var], mask))
    pred_dict['event_number'] = ak.to_numpy(arr['event_number']).astype(int)

    # compute mass and energy

    flat_class = ak.flatten(pred_dict[f'{model_name}_class'], axis=None)

    mass_lookup = np.array([class_mass_dict[i] for i in range(len(class_mass_dict))]) # ensure order
    flat_mass = mass_lookup[ak.to_numpy(flat_class)]
    pred_mass = ak.unflatten(flat_mass, ak.num(pred_dict[f'{model_name}_class']))
    pred_dict[f'{model_name}_mass'] = pred_mass

    pred_p = pred_dict[f'{model_name}_pt'] * np.cosh(pred_dict[f'{model_name}_eta'])
    pred_dict[f'{model_name}_e'] = np.sqrt(pred_p**2 + pred_mass**2)
    pred_dict[f'{model_name}_charge'] = ak.values_astype(pred_dict[f'{model_name}_class'] <= 2, int)
    print(f"\033[96m  load_predictions total: {time.perf_counter() - _t0:.1f}s\033[0m")
    return pred_dict


def load_target(target_path, drop_res=True):
    tree = uproot.open(target_path)['EventTree']
    vars_to_load = ['particle_pt', 'particle_eta', 'particle_phi', 'particle_e', 'particle_pdgid', 'eventNumber']
    remap = {'eventNumber': 'event_number'}

    target_dict_tmp = {}
    for var in tqdm(vars_to_load, desc="Loading target (segmented)...", total=len(vars_to_load)):
        new_var = remap.get(var, var)
        target_dict_tmp[new_var] = tree[var].array(library='ak')

    # class mapping
    flat_pdgid = ak.flatten(target_dict_tmp['particle_pdgid'])
    flat_class = ak.Array([pdgid_class_dict[x] for x in flat_pdgid])
    unflatten_class = ak.unflatten(flat_class, ak.num(target_dict_tmp['particle_pdgid']))

    # filter out the residual particles
    if drop_res: # will be dafault, here it is just for debugging
        mask = unflatten_class <= 4
        for key, val in target_dict_tmp.items():
            if key == 'event_number':
                continue
            target_dict_tmp[key] = val[mask]

    unique_sorted_ev_num = np.sort(np.unique(target_dict_tmp['event_number']))
    target_dict = {}
    for key, val in tqdm(target_dict_tmp.items(), desc="Merging target...", total=len(target_dict_tmp)):
        target_dict[key] = []
        for ev_num in unique_sorted_ev_num:
            mask = target_dict_tmp['event_number'] == ev_num
            target_dict[key].append(np.hstack(val[mask]))
        target_dict[key] = np.array(target_dict[key], dtype=object)
    target_dict['event_number'] = unique_sorted_ev_num

    # compute mass
    flat_mass = ak.Array([class_mass_dict[x] for x in flat_class])
    target_dict['particle_mass'] = ak.unflatten(flat_mass, ak.num(target_dict['particle_pdgid']))

    return target_dict



_JET_GROUPS = [
    ['AntiKt4TruthJetsPt', 'AntiKt4TruthJetsEta', 'AntiKt4TruthJetsPhi', 'AntiKt4TruthJetsE'],
    ['AntiKt4EMPFlowJetsPt', 'AntiKt4EMPFlowJetsEta', 'AntiKt4EMPFlowJetsPhi', 'AntiKt4EMPFlowJetsE'], # , 'AntiKt4EMPFlowJetsNConstituents'],
    ['AntiKt4EMTopoJetsPt', 'AntiKt4EMTopoJetsEta', 'AntiKt4EMTopoJetsPhi', 'AntiKt4EMTopoJetsE'],
]
# Lead branch of each collection — presence of this key marks the collection as loaded.
_JET_LEADS = ('AntiKt4TruthJetsPt', 'AntiKt4EMPFlowJetsPt', 'AntiKt4EMTopoJetsPt')


def _load_jets_atlas(jets_path, target_keys, scale_E_pT=1, num_workers=32):
    """Load jet collections from a dedicated jets file (no particle-level info) and
    reorder/filter them to align exactly with `target_keys` — the truth events'
    unique (eventNumber, mcChannelNumber) keys.

    `jets_path` may be a single ROOT file, a glob pattern matching several, or a
    .txt list of ROOT files. The returned arrays have one entry per `target_key`,
    in the same order.
    """
    import glob
    import concurrent.futures

    if isinstance(jets_path, str) and jets_path.endswith('.txt'):
        with open(jets_path) as f:
            paths = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    else:
        paths = sorted(glob.glob(jets_path))
    if not paths:
        raise ValueError(f"No ROOT files matched jets_path: {jets_path}")

    # Probe first file to decide which optional jet collections exist.
    with uproot.open(paths[0]) as _probe:
        available = set(_probe['EventTree'].keys())

    expressions = ['eventNumber', 'mcChannelNumber']
    for grp in _JET_GROUPS:
        if all(b in available for b in grp):
            expressions.extend(grp)
        else:
            print(f"\033[93mWarning: not all branches for jet group {grp} found in {paths[0]}; skipping this group.\033[0m")

    files_with_tree = [f"{p}:EventTree" for p in paths]
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)
    print(f"\033[96mLoading jets from {len(paths)} file(s) with {num_workers} workers...\033[0m")
    arr = uproot.concatenate(
        files_with_tree,
        expressions=expressions,
        library='ak',
        decompression_executor=executor,
        interpretation_executor=executor,
    )

    jet_event_number = ak.to_numpy(arr['eventNumber']).astype(np.int64)
    jet_mc_channel_number = ak.to_numpy(arr['mcChannelNumber']).astype(np.int64)
    jet_keys = jet_event_number * np.int64(10**10) + jet_mc_channel_number

    # Map each unique key to its row in the jets arrays (first occurrence wins),
    # then build the reorder index so jets line up with the truth events.
    key_to_idx = {}
    for i, k in enumerate(jet_keys):
        key_to_idx.setdefault(int(k), i)
    missing = [int(k) for k in target_keys if int(k) not in key_to_idx]
    if missing:
        raise ValueError(
            f"{len(missing)} truth event keys not found in jets_path files "
            f"(e.g. {missing[:5]}); cannot align jet collections.")
    order = np.array([key_to_idx[int(k)] for k in target_keys], dtype=np.int64)

    jets = {}
    if 'AntiKt4TruthJetsPt' in arr.fields:
        jets.update({
            'AntiKt4TruthJetsPt': arr['AntiKt4TruthJetsPt'][order] * scale_E_pT,
            'AntiKt4TruthJetsEta': arr['AntiKt4TruthJetsEta'][order],
            'AntiKt4TruthJetsPhi': arr['AntiKt4TruthJetsPhi'][order],
            'AntiKt4TruthJetsE': arr['AntiKt4TruthJetsE'][order] * scale_E_pT,
        })
    if 'AntiKt4EMPFlowJetsPt' in arr.fields:
        jets.update({
            'AntiKt4EMPFlowJetsPt': arr['AntiKt4EMPFlowJetsPt'][order] * scale_E_pT,
            'AntiKt4EMPFlowJetsEta': arr['AntiKt4EMPFlowJetsEta'][order],
            'AntiKt4EMPFlowJetsPhi': arr['AntiKt4EMPFlowJetsPhi'][order],
            'AntiKt4EMPFlowJetsE': arr['AntiKt4EMPFlowJetsE'][order] * scale_E_pT,
            # 'AntiKt4EMPFlowJetsNConstituents': ak.num(arr['AntiKt4EMPFlowJetsConstituentID'], axis=2)[order],
        })
    if 'AntiKt4EMTopoJetsPt' in arr.fields:
        jets.update({
            'AntiKt4EMTopoJetsPt': arr['AntiKt4EMTopoJetsPt'][order] * scale_E_pT,
            'AntiKt4EMTopoJetsEta': arr['AntiKt4EMTopoJetsEta'][order],
            'AntiKt4EMTopoJetsPhi': arr['AntiKt4EMTopoJetsPhi'][order],
            'AntiKt4EMTopoJetsE': arr['AntiKt4EMTopoJetsE'][order] * scale_E_pT,
        })
    return jets


def _augment_with_jets(result, jets_path, scale_E_pT=1, num_workers=32):
    """If jet collections are absent from `result` and `jets_path` is given, load
    them from `jets_path` (aligned to the truth events) and fill them in."""
    if jets_path is None:
        return result
    if all(k in result for k in _JET_LEADS):
        return result  # jets already present from filepath / cache
    jets = _load_jets_atlas(
        jets_path, result['unique_event_key'], scale_E_pT=scale_E_pT, num_workers=num_workers)
    for k, v in jets.items():
        if k not in result:
            result[k] = v
    return result


def load_truth_atlas(filepath, topo=False, fiducial_cuts=False, num_workers=32,
                     use_cache=True, cache_path=None, jets_path=None):
    from pathlib import Path

    # Resolve cache_path. If not given and filepath is a .txt list, derive a default
    # next to it; otherwise honour the path as-is.
    is_txt = isinstance(filepath, str) and filepath.endswith('.txt')
    if cache_path is None and is_txt:
        txt_path = Path(filepath)
        cache_path = txt_path.with_name(
            f"{txt_path.stem}_topo{int(topo)}_fid{int(fiducial_cuts)}.parquet"
        )
    elif cache_path is not None:
        cache_path = Path(cache_path)

    # Load from cache if available. When filepath is a real .txt list the cache must
    # also be newer than it; otherwise (e.g. filepath="") an existing cache is used
    # directly, so the truth file is never opened.
    if use_cache and cache_path is not None and cache_path.exists():
        txt_is_file = is_txt and Path(filepath).exists()
        if (not txt_is_file) or cache_path.stat().st_mtime > Path(filepath).stat().st_mtime:
            print(f"\033[96mLoading truth from cache: {cache_path}\033[0m")
            rec = ak.from_parquet(cache_path)
            merged = {k: rec[k] for k in rec.fields}
            # Restore numpy types for the flat per-event ints used in joins.
            for k in ("event_number", "mc_channel_number", "unique_event_key"):
                if k in merged:
                    merged[k] = ak.to_numpy(merged[k]).astype(np.int64)
            return _augment_with_jets(merged, jets_path, num_workers=num_workers)

    # If filepath is a .txt list, use uproot.concatenate to load all files in parallel
    # (one round-trip through uproot's threaded I/O instead of N serial Python loops).
    if is_txt:
        with open(filepath) as f:
            paths = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        if not paths:
            raise ValueError(f"No ROOT files listed in {filepath}")

        scale_E_pT = 1
        pt_min_gev = 0.01
        abs_eta_max = 3
        print("\033[96m" + f"E, pT will be scaled by {scale_E_pT}" + "\033[0m")
        print("\033[96m" + f"Will apply pT > {pt_min_gev} GeV and |eta| < {abs_eta_max} cuts to truth" + "\033[0m")

        # Probe first file once to decide which optional branches exist.
        with uproot.open(paths[0]) as _probe:
            available = set(_probe['EventTree'].keys())

        expressions = [
            'truthPartPt', 'truthPartE', 'truthPartEta', 'truthPartPhi',
            'truthPartPdgId', 'truthPartStatus',
            'eventNumber', 'mcChannelNumber',
        ]
        optional_groups = [
            ['AntiKt4TruthJetsPt', 'AntiKt4TruthJetsEta', 'AntiKt4TruthJetsPhi', 'AntiKt4TruthJetsE'],
            ['AntiKt4EMPFlowJetsPt', 'AntiKt4EMPFlowJetsEta', 'AntiKt4EMPFlowJetsPhi', 'AntiKt4EMPFlowJetsE'],# 'AntiKt4EMPFlowJetsConstituentID'],
            ['AntiKt4EMTopoJetsPt', 'AntiKt4EMTopoJetsEta', 'AntiKt4EMTopoJetsPhi', 'AntiKt4EMTopoJetsE'],
        ]
        for grp in optional_groups:
            if all(b in available for b in grp):
                expressions.extend(grp)
        if topo:
            expressions += ['cluster_E', 'cluster_Eta', 'cluster_Phi', 'cluster_Pt']

        # uproot.concatenate with explicit thread executors → parallel decompression + interpretation.
        files_with_tree = [f"{p}:EventTree" for p in paths]
        import concurrent.futures
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)
        print(f"\033[96mLoading {len(paths)} files with {num_workers} workers...\033[0m")
        arr = uproot.concatenate(
            files_with_tree,
            expressions=expressions,
            library='ak',
            decompression_executor=executor,
            interpretation_executor=executor,
        )

        # particle class (vectorized lookup on flattened pdgid)
        particle_pdgid = arr['truthPartPdgId']
        flat_pdgid = ak.to_numpy(ak.flatten(particle_pdgid))
        vectorized_lookup = np.vectorize(pdgid_class_dict.get, otypes=[int])
        flat_class_raw = vectorized_lookup(flat_pdgid, 0)
        particle_class_raw = ak.unflatten(ak.Array(flat_class_raw), ak.num(particle_pdgid))
        print("\033[91mskipping particle_class_mod computation (slow). It will be same as particle_class_raw\033[0m")
        particle_class_mod = deepcopy(particle_class_raw)

        particle_pt = arr['truthPartPt'] * scale_E_pT
        particle_e = arr['truthPartE'] * scale_E_pT
        particle_eta = arr['truthPartEta']
        particle_phi = arr['truthPartPhi']
        particle_gen_status = arr['truthPartStatus']

        if fiducial_cuts:
            mask_kin = (particle_pt >= pt_min_gev) * (abs(particle_eta) < abs_eta_max)
            mask_pdgid = (np.abs(particle_pdgid) != 12) & (np.abs(particle_pdgid) != 14) & (np.abs(particle_pdgid) != 16)
            mask_gen_status = (particle_gen_status == 1)
            mask = mask_kin * mask_pdgid * mask_gen_status
            particle_pt = particle_pt[mask]
            particle_e = particle_e[mask]
            particle_eta = particle_eta[mask]
            particle_phi = particle_phi[mask]
            particle_pdgid = particle_pdgid[mask]
            particle_class_raw = particle_class_raw[mask]
            particle_class_mod = particle_class_mod[mask]
            particle_gen_status = particle_gen_status[mask]

        event_number = ak.to_numpy(arr['eventNumber']).astype(np.int64)
        mc_channel_number = ak.to_numpy(arr['mcChannelNumber']).astype(np.int64)
        unique_event_key = event_number * np.int64(10**10) + mc_channel_number

        merged = {
            "particle_pt": particle_pt, "particle_eta": particle_eta, "particle_phi": particle_phi, "particle_e": particle_e,
            "particle_class": particle_class_raw, "particle_class_mod": particle_class_mod,
            "particle_pdgid": particle_pdgid, "particle_gen_status": particle_gen_status,
            "event_number": event_number,
            "mc_channel_number": mc_channel_number,
            "unique_event_key": unique_event_key,
        }

        if 'AntiKt4TruthJetsPt' in arr.fields:
            merged.update({
                'AntiKt4TruthJetsPt': arr['AntiKt4TruthJetsPt'] * scale_E_pT,
                'AntiKt4TruthJetsEta': arr['AntiKt4TruthJetsEta'],
                'AntiKt4TruthJetsPhi': arr['AntiKt4TruthJetsPhi'],
                'AntiKt4TruthJetsE': arr['AntiKt4TruthJetsE'] * scale_E_pT,
            })
        if 'AntiKt4EMPFlowJetsPt' in arr.fields:
            merged.update({
                'AntiKt4EMPFlowJetsPt': arr['AntiKt4EMPFlowJetsPt'] * scale_E_pT,
                'AntiKt4EMPFlowJetsEta': arr['AntiKt4EMPFlowJetsEta'],
                'AntiKt4EMPFlowJetsPhi': arr['AntiKt4EMPFlowJetsPhi'],
                'AntiKt4EMPFlowJetsE': arr['AntiKt4EMPFlowJetsE'] * scale_E_pT,
                # 'AntiKt4EMPFlowJetsNConstituents': ak.to_numpy(ak.num(arr['AntiKt4EMPFlowJetsConstituentID'], axis=2)),
            })
        if 'AntiKt4EMTopoJetsPt' in arr.fields:
            merged.update({
                'AntiKt4EMTopoJetsPt': arr['AntiKt4EMTopoJetsPt'] * scale_E_pT,
                'AntiKt4EMTopoJetsEta': arr['AntiKt4EMTopoJetsEta'],
                'AntiKt4EMTopoJetsPhi': arr['AntiKt4EMTopoJetsPhi'],
                'AntiKt4EMTopoJetsE': arr['AntiKt4EMTopoJetsE'] * scale_E_pT,
            })
        if topo:
            merged.update({
                'topo_e': arr['cluster_E'] * scale_E_pT,
                'topo_eta': arr['cluster_Eta'],
                'topo_phi': arr['cluster_Phi'],
                'topo_pt': arr['cluster_Pt'],
            })

        if use_cache:
            print(f"\033[96mWriting truth cache to {cache_path}\033[0m")
            try:
                ak.to_parquet(ak.zip(merged, depth_limit=1), cache_path)
            except Exception as e:
                print(f"\033[93mFailed to write cache ({e}); continuing without it.\033[0m")
        return _augment_with_jets(merged, jets_path, scale_E_pT=scale_E_pT, num_workers=num_workers)

    scale_E_pT=1
    pt_min_gev=0.01
    abs_eta_max=3
    print("\033[96m" + f"E, pT will be scaled by {scale_E_pT}" + "\033[0m")
    print("\033[96m" + f"Will apply pT > {pt_min_gev} GeV and |eta| < {abs_eta_max} cuts to truth" + "\033[0m")
    with uproot.open(filepath) as file:
        tree = file['EventTree']

        particle_pt  = tree['truthPartPt'].array(library='ak') * scale_E_pT
        particle_e   = tree['truthPartE'].array(library='ak') * scale_E_pT
        particle_eta = tree['truthPartEta'].array(library='ak')
        particle_phi = tree['truthPartPhi'].array(library='ak')
        particle_pdgid = tree['truthPartPdgId'].array(library='ak')
        particle_gen_status = tree['truthPartStatus'].array(library='ak')

        # raw particle class (these are not necessarily the target)
        # WARNING: defaulting to charged hadron
        flat_particle_pdgid = ak.to_numpy(ak.flatten(particle_pdgid))
        vectorized_lookup = np.vectorize(pdgid_class_dict.get, otypes=[int])
        flat_particle_class_raw = vectorized_lookup(flat_particle_pdgid, 0)
        flat_particle_class_raw = ak.Array(flat_particle_class_raw)
        particle_class_raw = ak.unflatten(flat_particle_class_raw, ak.num(particle_pdgid))

        # tracks
        # track_pt = tree['trackPt'].array(library='ak') * scale_E_pT
        # track_eta = tree['trackEta'].array(library='ak')
        # track_phi = tree['trackPhi'].array(library='ak')
        # track_particle_idx = tree['trackTruthParticleIndex'].array(library='ak')
        # track_accepted = tree['trackAccepted'].array(library='ak')
        
        # modified particle class
        # 0(ch had)->3(neut had), 1(e)->4(gamma), 2(mu)->3(ch had)
        print("\033[91m" + 'skipping particle_class_mod computation (slow). It will be same as particle_class_raw' + "\033[0m")
        particle_class_mod = deepcopy(particle_class_raw)

        # flat_particle_class_mod = deepcopy(flat_particle_class_raw)
        
        # particle_track_idx = []
        # for i in range(len(track_particle_idx)):
        #     part_track_idx_ev = np.zeros(len(particle_pt[i]), dtype=int) - 1
        #     valid_idx = track_particle_idx[i] >= 0
        #     part_track_idx_ev[track_particle_idx[i][valid_idx]] = np.arange(len(track_particle_idx[i]))[valid_idx]
        #     particle_track_idx.append(part_track_idx_ev)
        # particle_track_idx = ak.Array(particle_track_idx)

        # flat_particle_track_idx = ak.flatten(particle_track_idx)
        # flat_trackless_particle_mask = ak.where(flat_particle_track_idx >= 0, False, True)

        # # trackless ch hads and es become nu hads and photons (+3)
        # flat_trackless_chhad_and_e_mask = flat_trackless_particle_mask & (flat_particle_class_mod <= 1)
        # flat_particle_class_mod = ak.where(
        #     flat_trackless_chhad_and_e_mask, flat_particle_class_mod + 3, flat_particle_class_mod)

        # # trackless muons become neutral hadrons
        # flat_trackless_muon_mask = flat_trackless_particle_mask & (flat_particle_class_mod == 2)
        # flat_particle_class_mod = ak.where(flat_trackless_muon_mask, 3, flat_particle_class_mod)

        # # Unflatten the arrays back to their original structure
        # particle_class_mod = ak.unflatten(flat_particle_class_mod, ak.num(particle_pdgid))

        # pflow
        # ppflow_pt     = tree['PflowPt'].array(library='ak') * scale_E_pT
        # ppflow_eta    = tree['PflowEta'].array(library='ak')
        # ppflow_phi    = tree['PflowPhi'].array(library='ak')
        # ppflow_mass   = tree['PflowMass'].array(library='ak') * scale_E_pT
        # ppflow_charge = tree['PflowCharge'].array(library='ak')
        # ppflow_e      = np.sqrt((ppflow_pt * np.cosh(ppflow_eta))**2 + ppflow_mass**2)

        # pflow class
        # assume all ppflow are either charged hadrons or photons
        # ppflow_class = ak.where(ppflow_charge == 0, 4, 0)

        # fiducial cuts on particles
        if fiducial_cuts:
            mask_kin = (particle_pt >= pt_min_gev) * (abs(particle_eta) < abs_eta_max)
            mask_pdgid = (np.abs(particle_pdgid) != 12) & (np.abs(particle_pdgid) != 14) & (np.abs(particle_pdgid) != 16)
            mask_gen_status = (particle_gen_status == 1)
            mask = mask_kin * mask_pdgid * mask_gen_status

            particle_pt         = particle_pt[mask]
            particle_e          = particle_e[mask]
            particle_eta        = particle_eta[mask]
            particle_phi        = particle_phi[mask]
            particle_pdgid      = particle_pdgid[mask]
            particle_class_raw  = particle_class_raw[mask]
            particle_class_mod  = particle_class_mod[mask]
            particle_gen_status = particle_gen_status[mask]

            # # on tracks
            # for i in range(len(particle_class_mod)):

            #     # old idx to new idx
            #     masked_old_particle_idx = np.arange(len(mask[i]))[mask[i]]

            #     # on tracks (also make sure that the associated particle is in the fiducial region, otherwise we have a track with no particle)
            #     mask = (track_pt[i] >= pt_min_gev) * abs(track_eta[i]) < abs_eta_max
            #     track_part_pt_mask = np.in1d(track_particle_idx[i], masked_old_particle_idx)
            #     mask = mask * track_part_pt_mask
                
            #     track_pt[i]  = track_pt[i][mask]
            #     track_eta[i] = track_eta[i][mask]
            #     track_phi[i] = track_phi[i][mask]
            #     track_particle_idx[i] = np.array([
            #         np.where(masked_old_particle_idx == x)[0][0] for x in track_particle_idx[i][mask]
            #     ])

            # on ppflow (|eta| < 3)
            # mask = abs(ppflow_eta) < abs_eta_max
            # ppflow_pt = ppflow_pt[mask]
            # ppflow_eta = ppflow_eta[mask]
            # ppflow_phi = ppflow_phi[mask]
            # ppflow_mass = ppflow_mass[mask]
            # ppflow_charge = ppflow_charge[mask]
            # ppflow_class = ppflow_class[mask]
            # ppflow_e = ppflow_e[mask]

        # eventNumber alone is not unique across MC channels (JZ1..JZ4 etc.);
        # carry mcChannelNumber alongside and encode the pair as a single int64
        # unique key for downstream joins. mcChannelNumber maxes at ~7 digits;
        # 10**10 leaves headroom either way.
        event_number = tree['eventNumber'].array(library='np').astype(np.int64)
        mc_channel_number = tree['mcChannelNumber'].array(library='np').astype(np.int64)
        unique_event_key = event_number * np.int64(10**10) + mc_channel_number

        return_dict = {
            "particle_pt": particle_pt, "particle_eta": particle_eta, "particle_phi": particle_phi, "particle_e": particle_e,
            "particle_class": particle_class_raw, "particle_class_mod": particle_class_mod,
            "particle_pdgid": particle_pdgid, "particle_gen_status": particle_gen_status,
            # "track_pt": track_pt, "track_eta": track_eta, "track_phi": track_phi, "track_accepted": track_accepted, "track_particle_idx": track_particle_idx,
            # "ppflow_pt": ppflow_pt, "ppflow_eta": ppflow_eta, "ppflow_phi": ppflow_phi,
            # "ppflow_e": ppflow_e, "ppflow_mass": ppflow_mass, "ppflow_charge": ppflow_charge, "ppflow_class": ppflow_class,
            "event_number": event_number,
            "mc_channel_number": mc_channel_number,
            "unique_event_key": unique_event_key,
        }

        if 'AntiKt4TruthJetsPt' in tree.keys():
            return_dict.update({
                'AntiKt4TruthJetsPt': tree['AntiKt4TruthJetsPt'].array(library='ak') * scale_E_pT,
                'AntiKt4TruthJetsEta': tree['AntiKt4TruthJetsEta'].array(library='ak'),
                'AntiKt4TruthJetsPhi': tree['AntiKt4TruthJetsPhi'].array(library='ak'),
                'AntiKt4TruthJetsE': tree['AntiKt4TruthJetsE'].array(library='ak') * scale_E_pT
            })

        if 'AntiKt4EMPFlowJetsPt' in tree.keys():
            return_dict.update({
                'AntiKt4EMPFlowJetsPt': tree['AntiKt4EMPFlowJetsPt'].array(library='ak') * scale_E_pT,
                'AntiKt4EMPFlowJetsEta': tree['AntiKt4EMPFlowJetsEta'].array(library='ak'),
                'AntiKt4EMPFlowJetsPhi': tree['AntiKt4EMPFlowJetsPhi'].array(library='ak'),
                'AntiKt4EMPFlowJetsE': tree['AntiKt4EMPFlowJetsE'].array(library='ak') * scale_E_pT,
                # 'AntiKt4EMPFlowJetsNConstituents': np.array([
                #       np.array([len(x) for x in constids_ev]) for constids_ev in tree['AntiKt4EMPFlowJetsConstituentID'].array(library='np')
                # ], dtype=object)
            })

        if 'AntiKt4EMTopoJetsPt' in tree.keys():
            return_dict.update({
                'AntiKt4EMTopoJetsPt': tree['AntiKt4EMTopoJetsPt'].array(library='ak') * scale_E_pT,
                'AntiKt4EMTopoJetsEta': tree['AntiKt4EMTopoJetsEta'].array(library='ak'),
                'AntiKt4EMTopoJetsPhi': tree['AntiKt4EMTopoJetsPhi'].array(library='ak'),
                'AntiKt4EMTopoJetsE': tree['AntiKt4EMTopoJetsE'].array(library='ak') * scale_E_pT,
            })

        if topo:
            topo_e   = tree['cluster_E'].array(library='ak') * scale_E_pT
            topo_eta = tree['cluster_Eta'].array(library='ak')
            topo_phi = tree['cluster_Phi'].array(library='ak')
            topo_pt  = tree['cluster_Pt'].array(library='ak')

            return_dict.update({
                "topo_e": topo_e, "topo_eta": topo_eta, "topo_phi": topo_phi, "topo_pt": topo_pt #, "topo_e_lc": topo_e_lc
            })

    # sorted_idx = np.argsort(event_number)

    # for key in tqdm(return_dict.keys(), desc="Sorting truth events by event number..."):
    #     return_dict[key] = return_dict[key][sorted_idx]

    return_dict = _augment_with_jets(return_dict, jets_path, scale_E_pT=scale_E_pT, num_workers=num_workers)

    return return_dict