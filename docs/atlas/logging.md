# ATLAS Logging

This document summarizes how logging is wired for the ATLAS workflow in this
checkout. It is intended for new users configuring a training run from
`src/hepattn/experiments/atlas`.

## Source Files

The ATLAS training entrypoint is:

```text
src/hepattn/experiments/atlas/main.py
```

It uses the project CLI in `src/hepattn/utils/cli.py` with:

- model class: `hepattn.experiments.atlas.lightning_module.MPflow`
- data module class: `hepattn.experiments.atlas.pflow_data.PflowDataModule`
- default fit config: `src/hepattn/experiments/atlas/configs/base.yaml`

The logging-specific source files are:

- `src/hepattn/experiments/atlas/configs/base.yaml`
- `src/hepattn/experiments/atlas/configs/base_zj1234.yaml`
- `src/hepattn/experiments/atlas/configs/base_inference.yaml`
- `src/hepattn/experiments/atlas/logger_utils.py`
- `src/hepattn/utils/cli.py`
- `src/hepattn/callbacks/saveconfig.py`
- `src/hepattn/callbacks/checkpoint.py`
- `src/hepattn/experiments/atlas/predictionwriter.py`

## Default Logger

The checked-in ATLAS configs use Comet through the ATLAS wrapper logger:

```yaml
trainer:
  logger:
    class_path: hepattn.experiments.atlas.logger_utils.FixedCometLogger
    init_args:
      project_name: hepattn-atlas
```

Use `project_name`, not `project`, for this checkout. The pinned/local
Lightning logger source accepts `project_name`, and `docs/training.md` notes
that `project` can fail validation with some installed Lightning/Comet
versions.

`FixedCometLogger` subclasses Lightning's `CometLogger` and makes two ATLAS
compatibility changes:

1. If the CLI passes `save_dir`, it forwards it as Comet's
   `offline_directory`.
2. It forces the Comet experiment display name from `experiment_name` or
   `name`, avoiding Comet's generated adjective-style names.

The project depends on Comet directly through `pyproject.toml`:

```text
comet-ml>=3.47.3,<4
```

## New User Comet Setup

Configure credentials outside the repository. Do not put API keys or private
workspace details in committed YAML.

1. Create a Comet account at <https://www.comet.com>.
2. Create or choose the workspace and project where ATLAS runs should appear.
   The checked-in default project is `hepattn-atlas`.
3. On the machine or batch environment that will run training, configure the
   API key by one of these methods:

```bash
comet login
```

or:

```bash
export COMET_API_KEY=<your-api-key>
export COMET_WORKSPACE=<your-workspace>  # optional, useful for shared workspaces
```

The Comet SDK also honors `COMET_CONFIG` if the config file should live
somewhere other than the default home-directory location. This can matter on
batch systems where the job's home directory differs from the login shell.

For scheduler jobs, make sure the relevant environment variables are exported
into the job. The ATLAS PBS submit helper uses `qsub -v` only for checkpoint and
config variables, so Comet credentials must already be available in the job
environment or passed explicitly by the scheduler wrapper.

## Run Naming And Output Directories

The repo CLI adds a top-level `--name` option. It links that value into:

- `model.name`
- `trainer.logger.init_args.experiment_name`

During `fit`, the CLI rewrites `trainer.default_root_dir` to a timestamped run
directory:

```text
<default_root_dir>/<name>_YYYYMMDD-THHMMSS/
```

It also sets `trainer.logger.init_args.save_dir` to that timestamped directory
when a logger is enabled. For ATLAS, `FixedCometLogger` maps this to Comet's
offline directory argument.

Example:

```bash
cd src/hepattn/experiments/atlas

python main.py fit \
  -c configs/base.yaml \
  --name atlas_test_run \
  --trainer.default_root_dir /path/to/atlas/logs
```

This creates a run directory like:

```text
/path/to/atlas/logs/atlas_test_run_YYYYMMDD-THHMMSS/
```

## Local Artifacts

For `fit`, local outputs are written under the timestamped run directory:

```text
<run_dir>/config.yaml
<run_dir>/metadata.yaml
<run_dir>/ckpts/*.ckpt
```

`metadata.yaml` is written by `hepattn.callbacks.SaveConfig`. It records run
metadata such as:

- number of train and validation samples
- batch size
- trainable parameter count
- number of GPUs and GPU IDs
- dataloader worker count
- Torch, Lightning, and CUDA versions
- hostname
- logger save directory
- Comet URL, when available
- SLURM job ID, when `SLURM_JOB_ID` is set

When the trainer logger is a Comet logger, `SaveConfig` also uploads local YAML
files from the run directory as Comet assets and logs source files under
`src/**/*.py` as Comet code.

Checkpoints are written by `hepattn.callbacks.Checkpoint` under:

```text
<run_dir>/ckpts/
```

The filename includes epoch and monitored validation loss:

```text
epoch=003-val_loss=4.12345.ckpt
```

The checkpoint callback also calls Comet's `log_model` API by default. If a run
uses a non-Comet logger, disable this behavior or replace the callback before
expecting the run to work:

```yaml
trainer:
  callbacks:
    - class_path: hepattn.callbacks.Checkpoint
      init_args:
        monitor: val/loss
        log_model: false
```

## Metrics And Callback Logging

Model losses and metrics are logged through Lightning from
`src/hepattn/models/wrapper.py`. The ATLAS config also enables:

- `hepattn.callbacks.InferenceTimer`
- `hepattn.callbacks.SaveConfig`
- `hepattn.callbacks.Checkpoint`
- `hepattn.experiments.atlas.PflowPredictionWriter`
- `lightning.pytorch.callbacks.ModelSummary`
- `lightning.pytorch.callbacks.LearningRateMonitor`
- `lightning.pytorch.callbacks.TQDMProgressBar`

`PflowPredictionWriter` is active during `test`, not normal training. It writes
test outputs next to the checkpoint:

```text
<checkpoint_dir>/<checkpoint_name>__test.h5
<checkpoint_dir>/<checkpoint_name>__test.root
```

If `data.test_suff` is set, it is added to the test output filename.

## Disabling Logging

For smoke tests and debugging, disable online logging:

```bash
python main.py fit \
  -c configs/base.yaml \
  --trainer.logger=false \
  --trainer.fast_dev_run=true \
  --trainer.devices=1 \
  --data.num_train=100 \
  --data.num_val=50 \
  --data.batch_size=10 \
  --data.num_workers=0 \
  --trainer.callbacks=[]
```

The callback override matters because some callbacks expect logger or
checkpoint state that is not useful during a minimal `fast_dev_run`.

To disable only the logger for a fuller run:

```bash
python main.py fit -c configs/base.yaml --trainer.logger=false
```

If callbacks are left enabled while the logger is disabled, verify that the
checkpoint callback does not reach its Comet-specific `log_model` path.

## Testing And Inference Logging

During the CLI `test` subcommand, `src/hepattn/utils/cli.py` forcibly disables
the trainer logger:

```text
trainer.logger = False
```

It also disables the save-config callback, adjusts refresh-rate callbacks, and
forces testing to use one device. If `--ckpt_path` is omitted, it searches the
config directory's `ckpts/` folder and chooses the checkpoint with the lowest
loss parsed from the checkpoint filename.

For ATLAS inference-style evaluation, the usual command shape is:

```bash
python main.py test \
  -c configs/base_inference.yaml \
  -c configs/inference_override.yaml \
  --ckpt_path /path/to/checkpoint.ckpt
```

The inference wrapper scripts write process stdout/stderr to local files under
the ATLAS experiment `logs/` directory, separate from Comet.

## Batch Job Notes

The active ATLAS PBS helper is:

```text
src/hepattn/experiments/atlas/submit_training_atlas.py
```

It writes PBS stdout/stderr paths under `logs/` in the current working
directory, then runs:

```text
src/hepattn/experiments/atlas/run_on_node.sh
```

`run_on_node.sh` also writes a live tee log:

```text
logs/run_<jobid>.log
```

These files are scheduler/process logs, not Comet experiment logs.

The ATLAS directory also contains older SLURM submit scripts with Comet-related
echoes. They currently point at CLIC paths and should not be treated as the
source of truth for ATLAS logging.

## Using Another Logger

The trainer logger is a Lightning CLI object, so another Lightning logger can
be configured in principle. In this checkout, Comet is the only supported
online logger dependency.

WandB and MLflow are not declared dependencies in `pyproject.toml`, and they
are not installed in the local `.hepattn` environment inspected here. Switching
to either backend would require at least:

1. adding the dependency to `pyproject.toml`
2. changing the `trainer.logger.class_path`
3. changing logger init arguments to match that backend
4. handling Comet-specific callback behavior

The repo CLI currently links `--name` to
`trainer.logger.init_args.experiment_name`. That matches Comet, but not all
other loggers:

- WandB uses `name` for the run display name and `project` for the project.
- MLflow uses `run_name` for the run display name and `experiment_name` for the
  experiment.

The existing checkpoint callback also directly calls
`trainer.logger.experiment.log_model(...)`, which is Comet-shaped. For a
non-Comet logger, set `log_model: false` on `hepattn.callbacks.Checkpoint` or
replace the callback with a backend-specific implementation.

## Troubleshooting

If config parsing fails with a message like:

```text
Option 'project' is not accepted
Parser key "trainer.logger" does not validate
```

use `project_name` in the Comet logger config.

If Comet starts offline unexpectedly, check that `COMET_API_KEY` or the Comet
config file is visible inside the actual batch job environment. Remember that
the relevant process is the Python process running `main.py fit`, not just the
interactive shell where the job was submitted.

If the Comet UI shows generated experiment names, check that the config is
using `hepattn.experiments.atlas.logger_utils.FixedCometLogger` and that the
run has a meaningful top-level `name`.

If local run directories are created but no checkpoints appear, inspect the
trainer logs and callback state before cleaning outputs. The helper
`src/hepattn/utils/clean_logs.py` can delete run directories that contain only
YAML files and no `ckpts/` subdirectory, but it should be used carefully.

## References

- Local ATLAS training guide: `docs/training.md`
- Comet SDK configuration: <https://www.comet.com/docs/v2/guides/experiment-management/configure-sdk/>
- Comet PyTorch Lightning integration: <https://www.comet.com/docs/v2/integrations/ml-frameworks/pytorch-lightning/>
- Lightning CometLogger API: <https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.CometLogger.html>
