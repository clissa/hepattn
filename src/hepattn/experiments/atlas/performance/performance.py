import numpy as np
import awkward as ak
from tqdm import tqdm
from .jet_helper import JetHelper, compute_jets
from .cheap_jet import CheapJet
from .reader import load_predictions, load_target, load_truth_atlas
from scipy.optimize import linear_sum_assignment
from .utils import deltaR, delta_r


class PerformanceATLAS:
    
    def __init__(self, truth_path, pred_paths, ind_threshold, topo=False, proxy=False,
            target_path=None, fiducial_cuts_on_truth=False, load_hung_matched_truth=False, event_numbers=None,
            load_truth=True, num_workers=32, entry_stop=None):

        self.topo = topo
        self.proxy = proxy
    
        if isinstance(pred_paths, str):
            pred_paths = {'hgpflow': pred_paths}
        elif not isinstance(pred_paths, dict):
            raise ValueError("pred_paths should be string or dict of model_name: path")

        self.pred_dicts = {}
        for model, path in pred_paths.items():
            self.pred_dicts[model] = load_predictions(
                    path, threshold=ind_threshold, load_hung_matched_truth=load_hung_matched_truth,
                    model_name=model, num_workers=num_workers, entry_stop=entry_stop
                )

        self.target_dict = None
        if not target_path is None:
            self.target_dict = load_target(target_path)

        if isinstance(truth_path, dict):
            self.truth_dict = truth_path
        elif load_truth:
            self.truth_dict = load_truth_atlas(truth_path, topo=topo, fiducial_cuts=fiducial_cuts_on_truth)
        self.reorder_and_find_intersection(event_numbers)


    def reorder_and_find_intersection(self, event_numbers=None):

        ### check that event numbers are same in pred dicts
        pred_event_numbers = None
        for model_name, pred_dict in self.pred_dicts.items():
            if pred_event_numbers is None:
                pred_event_numbers = pred_dict['event_number']
            else:
                pred_event_numbers = np.intersect1d(pred_event_numbers, pred_dict['event_number'])

        self.common_event_numbers = np.intersect1d(
            self.truth_dict['event_number'], pred_event_numbers)
        if not self.target_dict is None:
            self.common_event_numbers = np.intersect1d(
                self.common_event_numbers, self.target_dict['event_number'])
        if event_numbers is not None:
            self.common_event_numbers = np.intersect1d(self.common_event_numbers, event_numbers)

        print('common event count:', len(self.common_event_numbers))

        # order them according to truth (we don't need to order self.truth_dict then)
        truth_mask = np.isin(self.truth_dict['event_number'], self.common_event_numbers)
        self.common_event_numbers = self.truth_dict['event_number'][truth_mask]
        
        # filter truth
        mask = np.isin(self.truth_dict['event_number'], self.common_event_numbers)
        if not mask.all():
            for var in tqdm(self.truth_dict.keys(), desc="Filtering truth...", total=len(self.truth_dict.keys())):
                self.truth_dict[var] = self.truth_dict[var][mask]

        # filter and reorder predictions
        for model_name, pred_dict in self.pred_dicts.items():
            positions = np.array([
                np.where(pred_dict['event_number'] == x)[0][0] for x in self.common_event_numbers]).astype(int)
            for var in tqdm(pred_dict.keys(), desc=f"Filtering and reordering {model_name} predictions...", total=len(pred_dict.keys())):
                pred_dict[var] = pred_dict[var][positions]
        # filter and reorder targets
        if not self.target_dict is None:
            positions = np.array([
                np.where(self.target_dict['event_number'] == x)[0][0] for x in self.common_event_numbers]).astype(int)
            for var in tqdm(self.target_dict.keys(), desc="Filtering and reordering targets...", total=len(self.target_dict.keys())):
                self.target_dict[var] = self.target_dict[var][positions]


    def compute_jets(self, radius=0.4, algo='antikt', n_procs=0, predictions_only=False, add_keys=None):
        jet_obj = JetHelper(radius=radius, algo=algo)
        
        if not predictions_only:
            # print('truth')
            # truth_mask = (self.truth_dict['particle_gen_status'] == 1)
            # self.truth_dict['truth_jets'] = compute_jets(jet_obj, 
            #     self.truth_dict['particle_pt'][truth_mask], self.truth_dict['particle_eta'][truth_mask],
            #     self.truth_dict['particle_phi'][truth_mask], self.truth_dict['particle_e'][truth_mask], 
            #     fourth_name='E', n_procs=n_procs)
            
            if 'AntiKt4TruthJetsPt' in self.truth_dict:
                print('AntiKt4TruthJets')
                self.truth_dict['AntiKt4TruthJets'] = []
                iter_obj = zip(self.truth_dict['AntiKt4TruthJetsPt'], self.truth_dict['AntiKt4TruthJetsEta'],
                    self.truth_dict['AntiKt4TruthJetsPhi'], self.truth_dict['AntiKt4TruthJetsE'])
                for pts, etas, phis, es in iter_obj:
                    cheap_jets_ev = [CheapJet.alternate(pt, eta, phi, e=e, n_const=0) for pt, eta, phi, e in zip(pts, etas, phis, es)]
                    self.truth_dict['AntiKt4TruthJets'].append(cheap_jets_ev)
            
            if 'AntiKt4EMPFlowJetsPt' in self.truth_dict:
                print('AntiKt4EMPFlowJets')
                self.truth_dict['AntiKt4EMPFlowJets'] = []
                iter_obj = zip(self.truth_dict['AntiKt4EMPFlowJetsPt'], self.truth_dict['AntiKt4EMPFlowJetsEta'],
                    self.truth_dict['AntiKt4EMPFlowJetsPhi'], self.truth_dict['AntiKt4EMPFlowJetsE']) #, self.truth_dict['AntiKt4EMPFlowJetsNConstituents'])
                # for pts, etas, phis, es, nconsts in iter_obj:
                for pts, etas, phis, es in iter_obj:
                    # cheap_jets_ev = [CheapJet.alternate(pt, eta, phi, e=e, n_const=nconst) for pt, eta, phi, e, nconst in zip(pts, etas, phis, es, nconsts)]
                    cheap_jets_ev = [CheapJet.alternate(pt, eta, phi, e=e, n_const=0) for pt, eta, phi, e in zip(pts, etas, phis, es)]
                    self.truth_dict['AntiKt4EMPFlowJets'].append(cheap_jets_ev)

            if 'AntiKt4EMTopoJetsPt' in self.truth_dict:
                print('AntiKt4EMTopoJets')
                self.truth_dict['AntiKt4EMTopoJets'] = []
                iter_obj = zip(self.truth_dict['AntiKt4EMTopoJetsPt'], self.truth_dict['AntiKt4EMTopoJetsEta'],
                    self.truth_dict['AntiKt4EMTopoJetsPhi'], self.truth_dict['AntiKt4EMTopoJetsE'])
                for pts, etas, phis, es in iter_obj:
                    cheap_jets_ev = [CheapJet.alternate(pt, eta, phi, e=e, n_const=0) for pt, eta, phi, e in zip(pts, etas, phis, es)]
                    self.truth_dict['AntiKt4EMTopoJets'].append(cheap_jets_ev)
            
        for model_name, pred_dict in self.pred_dicts.items():
            print(model_name)

            pred_dict['jets'] = compute_jets(jet_obj, 
                pred_dict[f'{model_name}_pt'], pred_dict[f'{model_name}_eta'],
                pred_dict[f'{model_name}_phi'], pred_dict[f'{model_name}_mass'],
                fourth_name='mass', n_procs=n_procs)

            if self.proxy:
                print(f'{model_name} proxy')
                pred_dict['proxy_jets'] = compute_jets(jet_obj,
                    pred_dict['proxy_pt'], pred_dict['proxy_eta'],
                    pred_dict['proxy_phi'], pred_dict[f'{model_name}_mass'],
                    fourth_name='mass', n_procs=n_procs)
                
            if not add_keys is None:
                for key in add_keys:
                    actual_key = f'{model_name}_{key}'
                    print(f'{model_name} {key}')
                    pred_dict[key + '_jets'] = compute_jets(jet_obj,
                        pred_dict[f'{actual_key}_pt'], pred_dict[f'{actual_key}_eta'],
                        pred_dict[f'{actual_key}_phi'], pred_dict[f'{model_name}_mass'],
                        fourth_name='mass', n_procs=n_procs)

        if not self.target_dict is None:
            print('target')
            self.target_dict['jets'] = compute_jets(jet_obj, 
                self.target_dict['particle_pt'], self.target_dict['particle_eta'],
                self.target_dict['particle_phi'], self.target_dict['particle_mass'],
                fourth_name='mass', n_procs=n_procs)

        if self.topo:
            print('topo')
            self.truth_dict['topo_jets'] = compute_jets(jet_obj, 
                self.truth_dict['topo_pt'], self.truth_dict['topo_eta'],
                self.truth_dict['topo_phi'], self.truth_dict['topo_e'], 
                fourth_name='E', n_procs=n_procs)


    def match_jets_single_ev(self, ref_jets, comp_jets):
        n_ref_jets = len(ref_jets)
        n_comp_jets = len(comp_jets)

        if n_ref_jets == 0 or n_comp_jets == 0:
            return [[],[]]
        
        dR_matrix = np.zeros((n_ref_jets, n_comp_jets))
        for i in range(n_ref_jets):
            for j in range(n_comp_jets):
                dR_matrix[i, j] = ref_jets[i].delta_R(comp_jets[j])

        row_indices, col_indices = linear_sum_assignment(dR_matrix, maximize=False)
        ref_jets_matched = [ref_jets[i] for i in row_indices]
        comp_jets_matched = [comp_jets[i] for i in col_indices]

        # sort both by pt of ref_jet
        sorted_idx = np.argsort([j.pt for j in ref_jets_matched])[::-1]
        ref_jets_matched = [ref_jets_matched[i] for i in sorted_idx]
        comp_jets_matched = [comp_jets_matched[i] for i in sorted_idx]

        return ref_jets_matched, comp_jets_matched


    def match_jets_all_ev(self, ref_jets, comp_jets):
        ref_jets_matched, comp_jets_matched = [], []
        for ev_i, (ref_jets_ev, comp_jets_ev) in enumerate(tqdm(zip(ref_jets, comp_jets), total=len(ref_jets), desc='Matching jets...')):
            ref_jets_ev_matched, comp_jets_ev_matched = self.match_jets_single_ev(ref_jets_ev, comp_jets_ev)
            ref_jets_matched.append(ref_jets_ev_matched)
            comp_jets_matched.append(comp_jets_ev_matched)

        return ref_jets_matched, comp_jets_matched


    def match_jets(self):
        # if 'AntiKt4TruthJets' in self.truth_dict:
        #     self.truth_dict['matched_truth_jets'] = self.match_jets_all_ev(
        #         self.truth_dict['AntiKt4TruthJets'], self.truth_dict['truth_jets'])
        
        if 'AntiKt4EMPFlowJets' in self.truth_dict:
            self.truth_dict['matched_AntiKt4EMPFlowJets'] = self.match_jets_all_ev(
                self.truth_dict['AntiKt4TruthJets'], self.truth_dict['AntiKt4EMPFlowJets'])

        if 'AntiKt4EMTopoJets' in self.truth_dict:
            self.truth_dict['matched_AntiKt4EMTopoJets'] = self.match_jets_all_ev(
                self.truth_dict['AntiKt4TruthJets'], self.truth_dict['AntiKt4EMTopoJets'])
        
        for model_name, pred_dict in self.pred_dicts.items():
            pred_dict[f'matched_{model_name}_jets'] = self.match_jets_all_ev(
                # self.truth_dict['truth_jets'], pred_dict['jets'])
                self.truth_dict['AntiKt4TruthJets'], pred_dict['jets'])
            
            if self.proxy:
                pred_dict['matched_proxy_jets'] = self.match_jets_all_ev(
                    # self.truth_dict['truth_jets'], pred_dict['proxy_jets'])
                    self.truth_dict['AntiKt4TruthJets'], pred_dict['proxy_jets'])

        if not self.target_dict is None:
            self.target_dict[f'matched_{self.model_name}_target_jets'] = self.match_jets_all_ev(
                self.truth_dict['AntiKt4TruthJets'], self.target_dict['jets'])

        if self.topo:
            self.truth_dict['matched_topo_jets'] = self.match_jets_all_ev(
                self.truth_dict['AntiKt4TruthJets'], self.truth_dict['topo_jets'])
            

    def hung_match_ev(self, ref_particles, comp_particles, return_unmatched=False, dR_threshold=None):
        ref_pt, ref_eta, ref_phi, ref_cl = ref_particles
        comp_pt, comp_eta, comp_phi, comp_cl = comp_particles

        cost_delpt_sq = (
            np.expand_dims(ref_pt, axis=1) - np.expand_dims(comp_pt, axis=0))**2
        cost_delpt_sq_by_pt_sq = cost_delpt_sq / np.expand_dims(ref_pt, axis=1)**2
        cost_deltaR = delta_r( # deltaR(
            np.expand_dims(ref_eta, axis=1), np.expand_dims(comp_eta, axis=0),
            np.expand_dims(ref_phi, axis=1), np.expand_dims(comp_phi, axis=0))
        cost = np.sqrt(cost_delpt_sq_by_pt_sq + cost_deltaR**2)

        ref_ch_mask = (ref_cl <= 2); comp_ch_mask = (comp_cl <= 2)

        # charged
        masked_cost = cost[np.ix_(ref_ch_mask, comp_ch_mask)]
        row_i, col_i = linear_sum_assignment(masked_cost, maximize=False)
        row_indices = np.arange(len(ref_pt))[ref_ch_mask][row_i]
        col_indices = np.arange(len(comp_pt))[comp_ch_mask][col_i]

        # neutral
        masked_cost = cost[np.ix_(~ref_ch_mask, ~comp_ch_mask)]
        row_i, col_i = linear_sum_assignment(masked_cost, maximize=False)
        row_indices = np.concatenate([row_indices, np.arange(len(ref_pt))[~ref_ch_mask][row_i]])
        col_indices = np.concatenate([col_indices, np.arange(len(comp_pt))[~comp_ch_mask][col_i]])

        if not dR_threshold is None:
            # apply hard cut on dR to flag unacceptable matches
            valid_match_mask = cost_deltaR[row_indices, col_indices] < dR_threshold
            row_indices = row_indices[valid_match_mask]
            col_indices = col_indices[valid_match_mask]

        ref_matched_dict = {
            'pt': ref_pt[row_indices], 'eta': ref_eta[row_indices],
            'phi': ref_phi[row_indices], 'class': ref_cl[row_indices]}
        comp_matched_dict = {
            'pt': comp_pt[col_indices], 'eta': comp_eta[col_indices],
            'phi': comp_phi[col_indices], 'class': comp_cl[col_indices]}

        ref_unmatched_dict = None; comp_unmatched_dict = None
        if return_unmatched:
            ref_unmatched_dict = {
                'pt': np.delete(ref_pt, row_indices), 'eta': np.delete(ref_eta, row_indices),
                'phi': np.delete(ref_phi, row_indices), 'class': np.delete(ref_cl, row_indices)}
            comp_unmatched_dict = {
                'pt': np.delete(comp_pt, col_indices), 'eta': np.delete(comp_eta, col_indices),
                'phi': np.delete(comp_phi, col_indices), 'class': np.delete(comp_cl, col_indices)}
        
        return ref_matched_dict, comp_matched_dict, ref_unmatched_dict, comp_unmatched_dict



    def hung_match_all_ev(self, ref_particles, comp_particles, flatten=False, return_unmatched=False, dR_threshold=None):
        rp_pt, rp_eta, rp_phi, rp_cl = ref_particles
        cp_pt, cp_eta, cp_phi, cp_cl = comp_particles

        print('ak->np comversion...', end=' ')
        rp_pt = [np.asarray(x) for x in rp_pt.tolist()]; rp_eta = [np.asarray(x) for x in rp_eta.tolist()]
        rp_phi = [np.asarray(x) for x in rp_phi.tolist()]; rp_cl = [np.asarray(x) for x in rp_cl.tolist()]

        cp_pt = [np.asarray(x) for x in cp_pt.tolist()]; cp_eta = [np.asarray(x) for x in cp_eta.tolist()]
        cp_phi = [np.asarray(x) for x in cp_phi.tolist()]; cp_cl = [np.asarray(x) for x in cp_cl.tolist()]
        print('done')

        ref_particles_matched  = {'pt': [], 'eta': [], 'phi': [], 'class': []}
        comp_particles_matched = {'pt': [], 'eta': [], 'phi': [], 'class': []}
        ref_particles_unmatched  = {'pt': [], 'eta': [], 'phi': [], 'class': []}
        comp_particles_unmatched = {'pt': [], 'eta': [], 'phi': [], 'class': []}

        for i in tqdm(range(len(ref_particles[0])), desc='Matching particles...'):
            ref_particles_ev_matched, comp_particles_ev_matched, \
            ref_particles_ev_unmatched, comp_particles_ev_unmatched = \
                self.hung_match_ev((rp_pt[i], rp_eta[i], rp_phi[i], rp_cl[i]),
                    (cp_pt[i], cp_eta[i], cp_phi[i], cp_cl[i]), 
                    return_unmatched=return_unmatched, dR_threshold=dR_threshold)
            
            for key in ref_particles_matched.keys():
                ref_particles_matched[key].append(ref_particles_ev_matched[key])
                comp_particles_matched[key].append(comp_particles_ev_matched[key])
                if return_unmatched:
                    ref_particles_unmatched[key].append(ref_particles_ev_unmatched[key])
                    comp_particles_unmatched[key].append(comp_particles_ev_unmatched[key])

        if flatten:
            for key in ref_particles_matched.keys():
                ref_particles_matched[key] = np.hstack(ref_particles_matched[key])
                comp_particles_matched[key] = np.hstack(comp_particles_matched[key])
                if return_unmatched:
                    ref_particles_unmatched[key] = np.hstack(ref_particles_unmatched[key])
                    comp_particles_unmatched[key] = np.hstack(comp_particles_unmatched[key])
        
        if return_unmatched:
            return ref_particles_matched, comp_particles_matched, ref_particles_unmatched, comp_particles_unmatched
        return ref_particles_matched, comp_particles_matched
    

    def hung_match_particles(self, flatten=False, return_unmatched=False, dR_threshold=None):

        for model_name, pred_dict in self.pred_dicts.items():
            pred_dict['matched_proxy_particles'] = self.hung_match_all_ev(
                (self.truth_dict['particle_pt'], self.truth_dict['particle_eta'], 
                self.truth_dict['particle_phi'], self.truth_dict['particle_class']),
                (pred_dict['proxy_pt'], pred_dict['proxy_eta'], 
                pred_dict['proxy_phi'], pred_dict[f'{model_name}_class']), 
                flatten, return_unmatched, dR_threshold=dR_threshold)
            pred_dict[f'matched_{model_name}_particles'] = self.hung_match_all_ev(
                (self.truth_dict['particle_pt'], self.truth_dict['particle_eta'], 
                self.truth_dict['particle_phi'], self.truth_dict['particle_class']),
                (pred_dict[f'{model_name}_pt'], pred_dict[f'{model_name}_eta'], 
                pred_dict[f'{model_name}_phi'], pred_dict[f'{model_name}_class']), 
                flatten, return_unmatched, dR_threshold=dR_threshold)
        if not self.target_dict is None:
            self.target_dict[f'matched_target_particles'] = self.hung_match_all_ev(
                (self.truth_dict['particle_pt'], self.truth_dict['particle_eta'], 
                 self.truth_dict['particle_phi'], self.truth_dict['particle_class']),
                (self.target_dict['particle_pt'], self.target_dict['particle_eta'], 
                 self.target_dict['particle_phi'], self.target_dict['particle_pdgid']), 
                 flatten, return_unmatched, dR_threshold=dR_threshold)
        
