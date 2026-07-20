"""Run ATLAS Run-4 inference and the corresponding performance notebook."""

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ATLAS_DIR = Path(__file__).resolve().parent
REPO_ROOT = ATLAS_DIR.parents[3]
NOTEBOOK_PATH = ATLAS_DIR / "notebooks" / "atlas_performance_run4.ipynb"


@dataclass(frozen=True)
class EvaluationPaths:
    """Resolved paths for one checkpoint evaluation."""

    checkpoint: Path
    config: Path
    inference_data: Path
    prediction: Path
    results: Path


def require_environment_path(name: str, environment: dict[str, str]) -> Path:
    """Return a required directory path from the environment.

    Raises:
        ValueError: If the variable is absent or empty.
        FileNotFoundError: If the variable does not name a directory.
    """
    value = environment.get(name)
    if not value:
        raise ValueError(f"{name} must be set")
    path = Path(value).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(f"{name} does not point to a directory: {path}")
    return path


def derive_paths(
    exp_name: str,
    run_name: str,
    ckpt_name: str,
    cfg_path: Path,
    environment: dict[str, str],
) -> EvaluationPaths:
    """Derive all evaluation paths and validate the required inputs.

    Raises:
        ValueError: If ``ckpt_name`` is not a checkpoint filename.
        FileNotFoundError: If a required file or directory is missing.
    """
    checkpoint_name = Path(ckpt_name)
    if checkpoint_name.name != ckpt_name or checkpoint_name.suffix != ".ckpt":
        raise ValueError("ckpt_name must be a .ckpt filename, not a path")

    checkpoint_base = require_environment_path("CKPT_PATH_TUNING", environment)
    partitions = require_environment_path("JZ_PATH_PARTITIONS", environment)
    checkpoint = checkpoint_base / exp_name / run_name / "ckpts" / checkpoint_name
    config = cfg_path.expanduser().resolve()
    if not config.is_file():
        raise FileNotFoundError(f"Config not found: {config}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    inference_data = partitions / "JZ1234_val.root"
    if not inference_data.is_file():
        raise FileNotFoundError(f"Inference data not found: {inference_data}")

    prediction = checkpoint.with_name(f"{checkpoint.stem}__test.root")
    results = REPO_ROOT / "results" / exp_name / run_name / checkpoint.stem
    return EvaluationPaths(
        checkpoint=checkpoint,
        config=config,
        inference_data=inference_data,
        prediction=prediction,
        results=results,
    )


def run_evaluation(exp_name: str, run_name: str, ckpt_name: str, cfg_path: Path) -> None:
    """Run inference, then execute the Run-4 performance notebook.

    Raises:
        FileNotFoundError: If inference or notebook execution does not write its expected output.
    """
    paths = derive_paths(exp_name, run_name, ckpt_name, cfg_path, dict(os.environ))
    environment = os.environ.copy()
    environment.update({
        "EXP_NAME": exp_name,
        "RUN_NAME": run_name,
        "PRED_NAME": paths.prediction.name,
        "RESULTS_PATH": str(paths.results),
    })

    print(f"Checkpoint: {paths.checkpoint}")
    print(f"Config: {paths.config}")
    print(f"Input ROOT: {paths.inference_data}")
    subprocess.run(
        [
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
            "--data.is_inference",
            "true",
        ],
        cwd=ATLAS_DIR,
        env=environment,
        check=True,
    )
    if not paths.prediction.is_file():
        raise FileNotFoundError(f"Inference completed without writing prediction: {paths.prediction}")

    paths.results.mkdir(parents=True, exist_ok=True)
    output_name = "atlas_performance_run4.executed"
    executed_notebook = paths.results / f"{output_name}.ipynb"
    print(f"Prediction: {paths.prediction}")
    print(f"Executing notebook: {NOTEBOOK_PATH}")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--output",
            output_name,
            "--output-dir",
            str(paths.results),
            "--ExecutePreprocessor.timeout=-1",
            str(NOTEBOOK_PATH),
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
    )
    if not executed_notebook.is_file():
        raise FileNotFoundError(f"Notebook execution completed without writing: {executed_notebook}")
    print(f"Executed notebook: {executed_notebook}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--ckpt-name", required=True)
    parser.add_argument("--cfg-path", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    """Run the command-line entry point."""
    args = parse_args()
    run_evaluation(args.exp_name, args.run_name, args.ckpt_name, args.cfg_path)


if __name__ == "__main__":
    main()
