from pathlib import Path

import numpy as np
import pytest

from hepattn.experiments.atlas import run_evaluation
from hepattn.experiments.atlas.performance import reader


def make_environment(tmp_path, split="val"):
    checkpoint_base = tmp_path / "checkpoints"
    partitions = tmp_path / "data" / "datasets"
    jets = partitions.parent / "edreyer"
    checkpoint_base.mkdir()
    partitions.mkdir(parents=True)
    jets.mkdir()
    (jets / "merged_JZ1_jets.root").touch()
    source = tmp_path / f"source_{split}.root"
    source.touch()
    data = partitions / f"JZ1234_{split}.root"
    data.touch()
    (partitions / f"{split}_raw_paths.txt").write_text(f"{source}\n")
    return {"CKPT_PATH_TUNING": str(checkpoint_base), "JZ_PATH_PARTITIONS": str(partitions)}, data


def make_paths(tmp_path, split="val"):
    env, data = make_environment(tmp_path, split)
    checkpoint = Path(env["CKPT_PATH_TUNING"]) / "experiment" / "run" / "ckpts" / "epoch=001.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    config = tmp_path / "config.yaml"
    config.touch()
    return env, data, checkpoint, config


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_derive_paths_maps_merged_split_to_truth_sources(tmp_path, monkeypatch, split):
    env, data, checkpoint, config = make_paths(tmp_path, split)
    repo = tmp_path / "repo"
    monkeypatch.setattr(run_evaluation, "REPO_ROOT", repo)

    paths = run_evaluation.derive_paths("experiment", "run", checkpoint.name, config, data, env)

    assert paths.checkpoint == checkpoint
    assert paths.inference_data == data
    assert paths.truth_sources == data.with_name(f"{split}_raw_paths.txt")
    assert paths.prediction == checkpoint.with_name("epoch=001__test.root")
    assert paths.results == repo / "results" / "experiment" / "run" / "epoch=001"
    assert paths.plots == paths.results / "plots_run4"
    assert paths.truth_jets.endswith("edreyer/merged_JZ*_jets.root")


def test_derive_paths_defaults_to_validation_data(tmp_path):
    env, data, checkpoint, config = make_paths(tmp_path)

    paths = run_evaluation.derive_paths("experiment", "run", checkpoint.name, config, None, env)

    assert paths.inference_data == data


@pytest.mark.parametrize("filename", ["custom.root", "JZ1234_other.root", "JZ1234_train.h5"])
def test_infer_truth_sources_rejects_unknown_merged_filename(tmp_path, filename):
    with pytest.raises(ValueError, match=r"JZ1234_train\.root"):
        run_evaluation.infer_truth_sources(tmp_path / filename)


def test_infer_truth_sources_rejects_empty_source_list(tmp_path):
    data = tmp_path / "JZ1234_train.root"
    data.touch()
    truth_list = tmp_path / "train_raw_paths.txt"
    truth_list.write_text("\n# comment\n")
    with pytest.raises(ValueError, match="empty"):
        run_evaluation.infer_truth_sources(data)


@pytest.mark.parametrize("value", [0, -2])
def test_validate_num_events_rejects_non_positive_values(value):
    with pytest.raises(argparse_error()):
        run_evaluation.validate_num_events(value)


def argparse_error():
    return run_evaluation.argparse.ArgumentTypeError


def test_build_inference_command_for_gpu(tmp_path):
    env, data, checkpoint, config = make_paths(tmp_path)
    paths = run_evaluation.derive_paths("experiment", "run", checkpoint.name, config, data, env)

    command = run_evaluation.build_inference_command(paths, 1000, "gpu")

    assert command[2:4] == ["main.py", "test"]
    assert command[command.index("--data.num_test") + 1] == "1000"
    assert command[command.index("--trainer.accelerator") + 1] == "gpu"
    assert command[command.index("--trainer.devices") + 1] == "1"
    assert "--trainer.precision" not in command
    assert "jupyter" not in command
    assert "nbconvert" not in command


def test_build_inference_command_for_cpu(tmp_path):
    env, data, checkpoint, config = make_paths(tmp_path)
    paths = run_evaluation.derive_paths("experiment", "run", checkpoint.name, config, data, env)

    command = run_evaluation.build_inference_command(paths, -1, "cpu")

    assert command[command.index("--trainer.accelerator") + 1] == "cpu"
    assert command[command.index("--trainer.precision") + 1] == "32-true"
    attention = "--model.model.init_args.encoder.init_args.attn_type"
    assert command[command.index(attention) + 1] == "torch"


def test_parse_args_uses_editable_defaults_and_cli_overrides(monkeypatch):
    monkeypatch.setattr(run_evaluation, "EXP_NAME", "default-exp")
    monkeypatch.setattr(run_evaluation, "RUN_NAME", "default-run")
    monkeypatch.setattr(run_evaluation, "CKPT_NAME", "default.ckpt")
    monkeypatch.setattr(run_evaluation, "CFG_PATH", Path("default.yaml"))

    defaults = run_evaluation.parse_args([])
    overrides = run_evaluation.parse_args([
        "--exp-name",
        "other-exp",
        "--run-name",
        "other-run",
        "--ckpt-name",
        "other.ckpt",
        "--cfg-path",
        "other.yaml",
        "--data-path",
        "JZ1234_train.root",
        "--num-events",
        "1000",
        "--device",
        "cpu",
    ])

    assert (defaults.exp_name, defaults.run_name, defaults.ckpt_name) == ("default-exp", "default-run", "default.ckpt")
    assert overrides.exp_name == "other-exp"
    assert overrides.cfg_path == Path("other.yaml")
    assert overrides.data_path == Path("JZ1234_train.root")
    assert overrides.num_events == 1000
    assert overrides.device == "cpu"


def test_run_evaluation_stops_if_prediction_is_missing(tmp_path, monkeypatch):
    env, data, checkpoint, config = make_paths(tmp_path)
    monkeypatch.setattr(run_evaluation.os, "environ", env)
    monkeypatch.setattr(run_evaluation.subprocess, "run", lambda *_args, **_kwargs: None)
    reconstruction_called = False

    def fake_reconstruction(_paths, *_args):
        nonlocal reconstruction_called
        reconstruction_called = True

    monkeypatch.setattr(run_evaluation, "run_reconstruction_evaluation", fake_reconstruction)
    with pytest.raises(FileNotFoundError, match="without writing prediction"):
        run_evaluation.run_evaluation("experiment", "run", checkpoint.name, config, data, 1000, "gpu")
    assert not reconstruction_called


def test_run_evaluation_runs_reconstruction_after_prediction(tmp_path, monkeypatch):
    env, data, checkpoint, config = make_paths(tmp_path)
    monkeypatch.setattr(run_evaluation.os, "environ", env)
    calls = []

    def fake_inference(*_args, **_kwargs):
        calls.append("inference")
        checkpoint.with_name("epoch=001__test.root").touch()

    def fake_reconstruction(paths, *_args):
        calls.append(("reconstruction", paths.prediction))
        return 1000

    monkeypatch.setattr(run_evaluation.subprocess, "run", fake_inference)
    monkeypatch.setattr(run_evaluation, "run_reconstruction_evaluation", fake_reconstruction)

    run_evaluation.run_evaluation("experiment", "run", checkpoint.name, config, data, 1000, "gpu")

    assert calls == ["inference", ("reconstruction", checkpoint.with_name("epoch=001__test.root"))]


def test_derive_paths_requires_environment_directories(tmp_path):
    _env, _data, _checkpoint, config = make_paths(tmp_path)
    with pytest.raises(ValueError, match="CKPT_PATH_TUNING must be set"):
        run_evaluation.derive_paths("experiment", "run", "epoch=001.ckpt", config, None, {})


def test_expected_plot_names_match_current_notebook_outputs():
    assert run_evaluation.EXPECTED_PLOTS == (
        "plot_dijet_jet_residuals.png",
        "plot_dijet_jet_ratio_pt_boxplot.png",
        "plot_dijet_jet_ratio_eta_boxplot.png",
        "plot_dijet_jet_response_meanstd.png",
        "plot_dijet_jet_response_medianiqr.png",
        "dijet_eff_fr_purity.png",
        "dijet_particle_residuals.png",
        "dijet_particle_residuals_combined.png",
    )


def test_resolve_checkpoint_prefers_existing_tuning_copy(tmp_path, monkeypatch):
    tuning = tmp_path / "tuning"
    expected = tuning / "experiment" / "run" / "ckpts" / "epoch=001.ckpt"
    expected.parent.mkdir(parents=True)
    expected.write_text("tuning")
    repo = tmp_path / "repo"
    training = repo / "results" / "experiment" / "run" / "ckpts" / "epoch=001.ckpt"
    training.parent.mkdir(parents=True)
    training.write_text("training")
    monkeypatch.setattr(run_evaluation, "REPO_ROOT", repo)

    resolved = run_evaluation.resolve_checkpoint(tuning, "experiment", "run", Path("epoch=001.ckpt"))

    assert resolved == expected
    assert resolved.read_text() == "tuning"


def test_resolve_checkpoint_copies_training_checkpoint_to_tuning(tmp_path, monkeypatch):
    tuning = tmp_path / "tuning"
    repo = tmp_path / "repo"
    training = repo / "results" / "experiment" / "run" / "ckpts" / "epoch=001.ckpt"
    training.parent.mkdir(parents=True)
    training.write_text("checkpoint")
    monkeypatch.setattr(run_evaluation, "REPO_ROOT", repo)

    resolved = run_evaluation.resolve_checkpoint(tuning, "experiment", "run", Path("epoch=001.ckpt"))

    assert resolved == tuning / "experiment" / "run" / "ckpts" / "epoch=001.ckpt"
    assert resolved.read_text() == "checkpoint"


def test_resolve_checkpoint_fails_when_both_locations_are_missing(tmp_path, monkeypatch):
    tuning = tmp_path / "tuning"
    repo = tmp_path / "repo"
    monkeypatch.setattr(run_evaluation, "REPO_ROOT", repo)

    with pytest.raises(FileNotFoundError, match="tuning or training results"):
        run_evaluation.resolve_checkpoint(tuning, "experiment", "run", Path("epoch=001.ckpt"))


def test_resolve_truth_sources_clears_missing_basename_when_found_later(tmp_path, capsys):
    data = tmp_path / "JZ1234_train.root"
    data.touch()
    filename = "user.edreyer.50520198._000081.mltree.root"
    missing = tmp_path / "JZ1" / filename
    existing = tmp_path / "JZ3" / filename
    existing.parent.mkdir()
    existing.touch()
    source_list = tmp_path / "train_raw_paths.txt"
    source_list.write_text(f"{missing}\n{existing}\n")

    resolved_list = run_evaluation.resolve_truth_sources(data, tmp_path / "results")

    assert resolved_list.read_text().splitlines() == [str(existing)]
    assert "Skipping missing truth source" in capsys.readouterr().out


def test_resolve_truth_sources_does_not_mark_basename_missing_when_found_earlier(tmp_path):
    data = tmp_path / "JZ1234_train.root"
    data.touch()
    filename = "duplicate.root"
    existing = tmp_path / "JZ3" / filename
    missing = tmp_path / "JZ1" / filename
    existing.parent.mkdir()
    existing.touch()
    (tmp_path / "train_raw_paths.txt").write_text(f"{existing}\n{missing}\n")

    resolved_list = run_evaluation.resolve_truth_sources(data, tmp_path / "results")

    assert resolved_list.read_text().splitlines() == [str(existing)]


def test_resolve_truth_sources_reports_unresolved_basename_and_skips_path(tmp_path, capsys):
    data = tmp_path / "JZ1234_test.root"
    data.touch()
    existing = tmp_path / "existing.root"
    existing.touch()
    missing = tmp_path / "missing.root"
    (tmp_path / "test_raw_paths.txt").write_text(f"{existing}\n{missing}\n")

    resolved_list = run_evaluation.resolve_truth_sources(data, tmp_path / "results")

    assert resolved_list.read_text().splitlines() == [str(existing)]
    output = capsys.readouterr().out
    assert str(missing) in output
    assert "missing.root" in output


def test_resolve_truth_sources_returns_original_list_when_all_files_exist(tmp_path):
    data = tmp_path / "JZ1234_val.root"
    data.touch()
    source = tmp_path / "source.root"
    source.touch()
    source_list = tmp_path / "val_raw_paths.txt"
    source_list.write_text(f"{source}\n")

    resolved_list = run_evaluation.resolve_truth_sources(data, tmp_path / "results")

    assert resolved_list == source_list
    assert not (tmp_path / "results").exists()


def test_jet_alignment_is_strict_by_default():
    target_keys = np.array([10, 20, 30])

    with pytest.raises(ValueError, match="cannot align jet collections"):
        reader._jet_alignment(target_keys, {10: 2, 30: 0})  # noqa: SLF001


def test_jet_alignment_skips_missing_events_within_limit(capsys):
    target_keys = np.array([10, 20, 30, 40])

    order, keep = reader._jet_alignment(  # noqa: SLF001
        target_keys,
        {10: 2, 20: 0, 30: 1},
        allow_missing=True,
        max_missing_pct=25,
        num_jet_files=4,
    )

    assert order.tolist() == [2, 0, 1]
    assert keep.tolist() == [True, True, True, False]
    warning = capsys.readouterr().out
    assert "1 of 4 truth event keys (25.00%)" in warning
    assert "4 jets_path file(s)" in warning


def test_jet_alignment_blocks_when_missing_events_exceed_limit():
    target_keys = np.array([10, 20, 30, 40])

    with pytest.raises(ValueError, match=r"exceeds the allowed maximum of 10\.00%"):
        reader._jet_alignment(target_keys, {10: 0, 20: 1, 30: 2}, allow_missing=True)  # noqa: SLF001


def test_parse_args_accepts_new_modes_and_missing_jet_options():
    args = run_evaluation.parse_args([
        "--inference-only",
        "--allow-missing-truth-jets",
        "--max-missing-truth-jets-pct",
        "7.5",
    ])

    assert args.inference_only
    assert not args.reco_only
    assert args.allow_missing_truth_jets
    assert args.max_missing_truth_jets_pct == 7.5

    with pytest.raises(SystemExit):
        run_evaluation.parse_args(["--inference-only", "--reco-only"])


def test_run_evaluation_inference_only_skips_reconstruction(tmp_path, monkeypatch):
    env, data, checkpoint, config = make_paths(tmp_path)
    monkeypatch.setattr(run_evaluation.os, "environ", env)
    reconstruction_called = False

    def fake_inference(*_args, **_kwargs):
        checkpoint.with_name("epoch=001__test.root").touch()

    def fake_reconstruction(*_args):
        nonlocal reconstruction_called
        reconstruction_called = True

    monkeypatch.setattr(run_evaluation.subprocess, "run", fake_inference)
    monkeypatch.setattr(run_evaluation, "run_reconstruction_evaluation", fake_reconstruction)

    run_evaluation.run_evaluation("experiment", "run", checkpoint.name, config, data, 1000, "gpu", inference_only=True)

    assert not reconstruction_called


def test_run_evaluation_reco_only_skips_inference(tmp_path, monkeypatch):
    env, data, checkpoint, config = make_paths(tmp_path)
    checkpoint.with_name("epoch=001__test.root").touch()
    monkeypatch.setattr(run_evaluation.os, "environ", env)
    monkeypatch.setattr(run_evaluation.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("inference ran"))
    monkeypatch.setattr(run_evaluation, "run_reconstruction_evaluation", lambda *_args: 1000)

    run_evaluation.run_evaluation("experiment", "run", checkpoint.name, config, data, 1000, "gpu", reco_only=True)
