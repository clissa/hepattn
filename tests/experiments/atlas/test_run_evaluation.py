from pathlib import Path

import pytest

from hepattn.experiments.atlas import run_evaluation


def environment(tmp_path):
    checkpoint_base = tmp_path / "checkpoints"
    partitions = tmp_path / "partitions"
    checkpoint_base.mkdir()
    partitions.mkdir()
    (partitions / "JZ1234_val.root").touch()
    return {"CKPT_PATH_TUNING": str(checkpoint_base), "JZ_PATH_PARTITIONS": str(partitions)}


def test_derive_paths_uses_experiment_path_and_checkpoint_stem(tmp_path):
    env = environment(tmp_path)
    checkpoint = Path(env["CKPT_PATH_TUNING"]) / "glow_baseline" / "atlas_run4" / "run" / "ckpts" / "epoch=027.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    config = tmp_path / "config.yaml"
    config.touch()

    paths = run_evaluation.derive_paths("glow_baseline/atlas_run4", "run", "epoch=027.ckpt", config, env)

    assert paths.checkpoint == checkpoint
    assert paths.prediction == checkpoint.with_name("epoch=027__test.root")
    assert paths.results == run_evaluation.REPO_ROOT / "results" / "glow_baseline" / "atlas_run4" / "run" / "epoch=027"


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"CKPT_PATH_TUNING": ""}, "CKPT_PATH_TUNING must be set"),
        ({"JZ_PATH_PARTITIONS": ""}, "JZ_PATH_PARTITIONS must be set"),
    ],
)
def test_derive_paths_requires_environment_directories(tmp_path, updates, match):
    env = environment(tmp_path)
    env.update(updates)
    config = tmp_path / "config.yaml"
    config.touch()

    with pytest.raises(ValueError, match=match):
        run_evaluation.derive_paths("experiment", "run", "epoch=001.ckpt", config, env)


def test_derive_paths_rejects_missing_or_invalid_checkpoint(tmp_path):
    env = environment(tmp_path)
    config = tmp_path / "config.yaml"
    config.touch()

    with pytest.raises(ValueError, match="filename"):
        run_evaluation.derive_paths("experiment", "run", "ckpts/epoch=001.ckpt", config, env)
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        run_evaluation.derive_paths("experiment", "run", "epoch=001.ckpt", config, env)


def test_derive_paths_rejects_missing_config(tmp_path):
    env = environment(tmp_path)
    checkpoint = Path(env["CKPT_PATH_TUNING"]) / "experiment" / "run" / "ckpts" / "epoch=001.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()

    with pytest.raises(FileNotFoundError, match="Config not found"):
        run_evaluation.derive_paths("experiment", "run", "epoch=001.ckpt", tmp_path / "missing.yaml", env)


def test_run_evaluation_fails_when_inference_writes_no_prediction(tmp_path, monkeypatch):
    env = environment(tmp_path)
    checkpoint = Path(env["CKPT_PATH_TUNING"]) / "experiment" / "run" / "ckpts" / "epoch=001.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    config = tmp_path / "config.yaml"
    config.touch()
    monkeypatch.setattr(run_evaluation.os, "environ", env)
    calls = []
    monkeypatch.setattr(run_evaluation.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs)))

    with pytest.raises(FileNotFoundError, match="without writing prediction"):
        run_evaluation.run_evaluation("experiment", "run", "epoch=001.ckpt", config)

    assert len(calls) == 1
    assert calls[0][0][0][2:4] == ["main.py", "test"]


def test_run_evaluation_runs_notebook_after_prediction(tmp_path, monkeypatch):
    env = environment(tmp_path)
    checkpoint = Path(env["CKPT_PATH_TUNING"]) / "experiment" / "run" / "ckpts" / "epoch=001.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    config = tmp_path / "config.yaml"
    config.touch()
    repo = tmp_path / "repo"
    monkeypatch.setattr(run_evaluation, "REPO_ROOT", repo)
    monkeypatch.setattr(run_evaluation.os, "environ", env)
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if command[2] == "main.py":
            checkpoint.with_name("epoch=001__test.root").touch()
        else:
            output = repo / "results" / "experiment" / "run" / "epoch=001" / "atlas_performance_run4.executed.ipynb"
            output.parent.mkdir(parents=True)
            output.touch()

    monkeypatch.setattr(run_evaluation.subprocess, "run", fake_run)

    run_evaluation.run_evaluation("experiment", "run", "epoch=001.ckpt", config)

    assert len(calls) == 2
    assert calls[0][0][2:4] == ["main.py", "test"]
    assert calls[1][0][2:5] == ["jupyter", "nbconvert", "--to"]
    assert calls[1][1]["env"]["PRED_NAME"] == "epoch=001__test.root"
