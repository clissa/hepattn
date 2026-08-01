"""Run ATLAS inference followed by Run-4 reconstruction evaluation."""
# ruff: noqa: PLC0415

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

ATLAS_DIR = Path(__file__).resolve().parent
REPO_ROOT = ATLAS_DIR.parents[3]

# Editable defaults. Matching command-line arguments take precedence.
EXP_NAME = "MDN_focal"
# RUN_NAME = "atlas_MDN_jz1234_v0_nopart_reproduce_20260716-T223207" # adamw
# CKPT_NAME = "epoch=000-val_loss=2.05008-7124.ckpt"
# CFG_PATH = ATLAS_DIR / "configs" / "mdn_adamw_base.yaml"
RUN_NAME = "atlas_MDN_focal_20260730-T223903"
CKPT_NAME = "epoch=048-val_loss=4.91068.ckpt"
CFG_PATH = ATLAS_DIR / "configs" / "base_MDN_focal.yaml"

IND_THRESHOLD = 0.50
ASSOC_TRACK_PT_THRESHOLD = 100
EXPECTED_PLOTS = (
    "plot_dijet_jet_residuals.png",
    "plot_dijet_jet_ratio_pt_boxplot.png",
    "plot_dijet_jet_ratio_eta_boxplot.png",
    "plot_dijet_jet_response_meanstd.png",
    "plot_dijet_jet_response_medianiqr.png",
    "dijet_eff_fr_purity.png",
    "dijet_particle_residuals.png",
    "dijet_particle_residuals_combined.png",
)


@dataclass(frozen=True)
class EvaluationPaths:
    checkpoint: Path
    config: Path
    inference_data: Path
    truth_sources: Path
    truth_jets: str
    prediction: Path
    results: Path
    plots: Path


def require_environment_path(name: str, environment: dict[str, str]) -> Path:
    value = environment.get(name)
    if not value:
        raise ValueError(f"{name} must be set")
    path = Path(value).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(f"{name} does not point to a directory: {path}")
    return path


def validate_num_events(value: int | str) -> int:
    value = int(value)
    if value == -1 or value > 0:
        return value
    raise argparse.ArgumentTypeError("num-events must be -1 or a positive integer")


def infer_truth_sources(data_path: Path) -> Path:
    match = re.fullmatch(r"JZ1234_(train|val|test)\.root", data_path.name)
    if match is None:
        raise ValueError("data-path must be named JZ1234_train.root, JZ1234_val.root, or JZ1234_test.root")
    truth_sources = data_path.with_name(f"{match.group(1)}_raw_paths.txt")
    if not truth_sources.is_file():
        raise FileNotFoundError(f"Truth source list not found: {truth_sources}")
    sources = [line.strip() for line in truth_sources.read_text().splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if not sources:
        raise ValueError(f"Truth source list is empty: {truth_sources}")
    return truth_sources


def resolve_truth_sources(data_path: Path, results: Path) -> Path:
    source_list = infer_truth_sources(data_path)
    sources = [Path(line.strip()) for line in source_list.read_text().splitlines() if line.strip() and not line.lstrip().startswith("#")]
    existing_sources = []
    present_basenames = set()
    missing_by_basename = {}
    skipped_any = False

    for source in sources:
        basename = source.name
        if source.is_file():
            existing_sources.append(source)
            present_basenames.add(basename)
            missing_by_basename.pop(basename, None)
            continue

        skipped_any = True
        print(f"Skipping missing truth source: {source}")
        if basename not in present_basenames:
            missing_by_basename.setdefault(basename, source)

    if not skipped_any:
        return source_list
    if not existing_sources:
        raise FileNotFoundError(f"No existing truth source files remain after filtering {source_list}")

    if missing_by_basename:
        print("Truth source basenames not found in any existing entry: " + ", ".join(sorted(missing_by_basename)))

    results.mkdir(parents=True, exist_ok=True)
    resolved_list = results / f"{source_list.stem}_resolved.txt"
    resolved_list.write_text("\n".join(str(source) for source in existing_sources) + "\n")
    return resolved_list


def resolve_checkpoint(checkpoint_base: Path, exp_name: str, run_name: str, checkpoint_name: Path) -> Path:
    checkpoint = checkpoint_base / exp_name / run_name / "ckpts" / checkpoint_name
    if checkpoint.is_file():
        return checkpoint

    training_checkpoint = REPO_ROOT / "results" / exp_name / run_name / "ckpts" / checkpoint_name
    if not training_checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found in tuning or training results: {checkpoint}; {training_checkpoint}")

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    print(f"Copying checkpoint from training results: {training_checkpoint} -> {checkpoint}")
    shutil.copy2(training_checkpoint, checkpoint)
    return checkpoint


def derive_paths(exp_name, run_name, ckpt_name, cfg_path, data_path, environment):
    checkpoint_name = Path(ckpt_name)
    if checkpoint_name.name != ckpt_name or checkpoint_name.suffix != ".ckpt":
        raise ValueError("ckpt-name must be a .ckpt filename, not a path")
    checkpoint_base = require_environment_path("CKPT_PATH_TUNING", environment)
    partitions = require_environment_path("JZ_PATH_PARTITIONS", environment)
    checkpoint = resolve_checkpoint(checkpoint_base, exp_name, run_name, checkpoint_name)
    config = Path(cfg_path).expanduser().resolve()
    inference_data = (Path(data_path) if data_path else partitions / "JZ1234_val.root").expanduser().resolve()
    for label, path in (("Config", config), ("Inference data", inference_data)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    results = REPO_ROOT / "results" / exp_name / run_name / checkpoint.stem
    truth_sources = resolve_truth_sources(inference_data, results)
    jets_dir = partitions.parent / "edreyer"
    truth_jets = str(jets_dir / "merged_JZ*_jets.root")
    if not list(jets_dir.glob("merged_JZ*_jets.root")):
        raise FileNotFoundError(f"No truth jet files match: {truth_jets}")
    prediction = checkpoint.with_name(f"{checkpoint.stem}__test.root")
    return EvaluationPaths(checkpoint, config, inference_data, truth_sources, truth_jets, prediction, results, results / "plots_run4")


def build_inference_command(paths: EvaluationPaths, num_events: int, device: str) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "main.py",
        "test",
        "-c",
        str(paths.config),
        "--ckpt_path",
        str(paths.checkpoint),
        "--data.test_path",
        str(paths.inference_data),
        "--data.num_test",
        str(num_events),
        "--data.is_inference",
        "true",
        "--trainer.accelerator",
        device,
        "--trainer.devices",
        "1",
    ]
    if device == "cpu":
        command.extend(["--trainer.precision", "32-true", "--model.model.init_args.encoder.init_args.attn_type", "torch"])
    return command


def _style_sheet():
    labels = {
        "truth": "Truth",
        "emtopo": "EMTopo",
        "empflow": "EMPFlow",
        "proxy": "Proxy (track-sub)",
        "hybrid": "Proxy (calo-only)",
        "mpflow": "GLOW",
    }
    return {
        "LABELS": labels,
        "LABEL_LEN": {"truth": 11, "emtopo": 9, "empflow": 10, "proxy": 11, "hybrid": 10, "mpflow": 10},
        "LINE_STYLES": {"truth": "--", "emtopo": "--", "empflow": "-", "proxy": "--", "hybrid": "--", "mpflow": "-"},
        "MARKERS": {"truth": "x", "emtopo": "s", "empflow": "h", "proxy": "x", "hybrid": "x", "mpflow": "^"},
        "COLORS": {"truth": "black", "emtopo": "olivedrab", "empflow": "teal", "proxy": "blue", "hybrid": "purple", "mpflow": "darkorange"},
        "HISTTYPES": {"truth": "step", "emtopo": "bar", "empflow": "bar", "proxy": "step", "hybrid": "step", "mpflow": "step"},
        "ALPHAS": {"truth": 1.0, "emtopo": 0.5, "empflow": 0.5, "proxy": 1.0, "hybrid": 1.0, "mpflow": 1.0},
    }


def _track_substituted(is_charged, particles, tracks, tag):
    import awkward as ak

    use_track_pt = is_charged * (tracks["pt"] < ASSOC_TRACK_PT_THRESHOLD)
    return {
        f"{tag}_pt": ak.where(use_track_pt, tracks["pt"], particles["pt"]),
        f"{tag}_eta": ak.where(is_charged, tracks["eta"], particles["eta"]),
        f"{tag}_phi": ak.where(is_charged, tracks["phi"], particles["phi"]),
    }


def run_reconstruction_evaluation(paths: EvaluationPaths) -> int:
    import awkward as ak
    import matplotlib.pyplot as plt
    import numpy as np

    from hepattn.experiments.atlas.performance.jet_helper import JetHelper, compute_jets
    from hepattn.experiments.atlas.performance.performance import PerformanceATLAS
    from hepattn.experiments.atlas.performance.plot_helper_event import (
        compute_jet_residual_dict,
        plot_jet_ratio_boxplot,
        plot_jet_residuals,
        plot_jet_response,
    )
    from hepattn.experiments.atlas.performance.plot_helper_particle import plot_eff_fr_purity, plot_residuals, plot_residuals_neutrals
    from hepattn.experiments.atlas.performance.reader import load_truth_atlas

    paths.plots.mkdir(parents=True, exist_ok=True)
    truth = load_truth_atlas(str(paths.truth_sources), topo=False, fiducial_cuts=False, cache_path=None, jets_path=paths.truth_jets)
    perf = PerformanceATLAS(
        truth_path=truth,
        pred_paths={"mpflow": paths.prediction},
        ind_threshold=IND_THRESHOLD,
        topo=False,
        proxy=True,
        target_path=None,
        load_hung_matched_truth=False,
        fiducial_cuts_on_truth=False,
        load_truth=True,
        entry_stop=None,
        num_workers=32,
    )
    perf.compute_jets(n_procs=45)
    perf.match_jets()

    pred = perf.pred_dicts["mpflow"]
    charged = pred["proxy_is_charged"]
    tracks = {field: pred[f"proxy_ch_{field}"] for field in ("pt", "eta", "phi")}
    proxy = {field: pred[f"proxy_{field}"] for field in ("pt", "eta", "phi")}
    mpflow = {field: pred[f"mpflow_{field}"] for field in ("pt", "eta", "phi")}
    pred.update(_track_substituted(charged, mpflow, tracks, "mpflow_tracksub"))
    pred.update(_track_substituted(charged, proxy, tracks, "mpflow_proxy_tracksub"))

    jet_helper = JetHelper(radius=0.4, algo="antikt")
    for key in ("tracksub", "proxy_tracksub"):
        prefix = f"mpflow_{key}"
        pred[f"{key}_jets"] = compute_jets(
            jet_helper, pred[f"{prefix}_pt"], pred[f"{prefix}_eta"], pred[f"{prefix}_phi"], pred["mpflow_mass"], fourth_name="mass", n_procs=30
        )
    pred["mpflow_calo_jets"] = compute_jets(
        jet_helper,
        pred["proxy_neut_pt"],
        ak.where(charged, pred["proxy_ch_eta"], pred["proxy_neut_eta"]),
        ak.where(charged, pred["proxy_ch_phi"], pred["proxy_neut_phi"]),
        pred["mpflow_mass"],
        fourth_name="mass",
        n_procs=30,
    )
    for key in ("tracksub", "proxy_tracksub", "mpflow_calo"):
        pred[f"matched_{key}_jets"] = perf.match_jets_all_ev(perf.truth_dict["AntiKt4TruthJets"], pred[f"{key}_jets"])

    styles = _style_sheet()
    residuals = compute_jet_residual_dict(
        {
            "emtopo": perf.truth_dict["matched_AntiKt4EMTopoJets"],
            "empflow": perf.truth_dict["matched_AntiKt4EMPFlowJets"],
            "mpflow": pred["matched_mpflow_jets"],
            "proxy": pred["matched_proxy_tracksub_jets"],
            "hybrid": pred["matched_mpflow_calo_jets"],
        },
        dr_cut=0.1,
        leading_N_jets=2,
        pt_min=20,
    )
    pt_bins = np.array([20, 50, 100, 150, 200, 250, 300, 350, 400, 450])
    figures = {
        "plot_dijet_jet_residuals.png": plot_jet_residuals(residuals, pt_relative=True, stylesheet=styles, separate_figures=False),
        "plot_dijet_jet_ratio_pt_boxplot.png": plot_jet_ratio_boxplot(residuals, bins=pt_bins, stylesheet=styles),
        "plot_dijet_jet_ratio_eta_boxplot.png": plot_jet_ratio_boxplot(residuals, var="abseta", bins=np.linspace(0, 3, 6), stylesheet=styles),
    }
    responses = plot_jet_response(residuals, pt_bins=pt_bins, separate_figures=True, stylesheet=styles, ratio_panel=True)
    figures["plot_dijet_jet_response_meanstd.png"], figures["plot_dijet_jet_response_medianiqr.png"] = responses

    perf.hung_match_particles(flatten=True, return_unmatched=True, dR_threshold=0.5)
    matched = pred["matched_mpflow_particles"]
    figures["dijet_eff_fr_purity.png"] = plot_eff_fr_purity(
        {"mpflow": {"ref_matched": matched[0], "comp_matched": matched[1], "ref_unmatched": matched[2], "comp_unmatched": matched[3]}},
        stylesheet={
            "LABELS": {"mpflow": "GLOW"},
            "LINE_STYLES": {"mpflow": "--"},
            "COLORS": {"mpflow": {"neut had": "mediumseagreen", "photon": "tomato"}},
        },
    )
    particle_styles = deepcopy(styles)
    particle_styles["COLORS"]["proxy"] = "dodgerblue"
    particle_styles["LINE_STYLES"]["proxy"] = "--"
    particle_matches = {"mpflow": matched, "proxy": pred["matched_proxy_particles"]}
    figures["dijet_particle_residuals.png"] = plot_residuals(
        particle_matches,
        pt_relative=True,
        log_y=False,
        qs={"Charged": {"pt": 90, "eta": 80, "phi": 80}, "Neutral": {"pt": 90, "eta": 80, "phi": 80}},
        stylesheet=particle_styles,
    )
    figures["dijet_particle_residuals_combined.png"] = plot_residuals_neutrals(
        particle_matches,
        pt_relative=True,
        log_y=False,
        qs={"Neutral hadron": {"pt": 98, "eta": 75, "phi": 75}, "Photon": {"pt": 99, "eta": 90, "phi": 90}},
        stylesheet=particle_styles,
        separate_figures=False,
    )
    for filename, figure in figures.items():
        figure.savefig(paths.plots / filename, dpi=300, bbox_inches="tight")
        plt.close(figure)
    missing = [filename for filename in EXPECTED_PLOTS if not (paths.plots / filename).is_file()]
    if missing:
        raise FileNotFoundError(f"Evaluation completed without writing expected plots: {missing}")
    return len(perf.common_event_numbers)


def run_evaluation(exp_name, run_name, ckpt_name, cfg_path, data_path, num_events, device):
    paths = derive_paths(exp_name, run_name, ckpt_name, cfg_path, data_path, dict(os.environ))
    print(f"Checkpoint: {paths.checkpoint}")
    print(f"Config: {paths.config}")
    print(f"Input ROOT: {paths.inference_data}")
    print(f"Truth sources: {paths.truth_sources}")
    print(f"Device: {device}; events: {'all' if num_events == -1 else num_events}")
    started = time.perf_counter()
    subprocess.run(build_inference_command(paths, num_events, device), cwd=ATLAS_DIR, check=True)
    inference_seconds = time.perf_counter() - started
    if not paths.prediction.is_file():
        raise FileNotFoundError(f"Inference completed without writing prediction: {paths.prediction}")
    started = time.perf_counter()
    common_events = run_reconstruction_evaluation(paths)
    evaluation_seconds = time.perf_counter() - started
    print(f"Prediction: {paths.prediction}")
    print(f"Results: {paths.results}")
    print(f"Common evaluated events: {common_events:,}")
    print(f"Inference time: {inference_seconds:.1f} seconds")
    print(f"Reconstruction evaluation time: {evaluation_seconds:.1f} seconds")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", default=EXP_NAME)
    parser.add_argument("--run-name", default=RUN_NAME)
    parser.add_argument("--ckpt-name", default=CKPT_NAME)
    parser.add_argument("--cfg-path", default=CFG_PATH, type=Path)
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--num-events", default=-1, type=validate_num_events)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    args = parser.parse_args(argv)
    for name in ("exp_name", "run_name", "ckpt_name"):
        if not getattr(args, name):
            parser.error(f"--{name.replace('_', '-')} is required unless its editable default is set")
    return args


def main():
    args = parse_args()
    run_evaluation(args.exp_name, args.run_name, args.ckpt_name, args.cfg_path, args.data_path, args.num_events, args.device)


if __name__ == "__main__":
    main()
