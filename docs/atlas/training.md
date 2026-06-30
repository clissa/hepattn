# ATLAS Training

This document explains how to run the ATLAS GLOW training entrypoint and how to do a small dry-run before launching a full job.

## Entry Point

The ATLAS training script is:

```text
src/hepattn/experiments/atlas/main.py
```

It starts the repo's custom Lightning CLI with:

- model class: `hepattn.experiments.atlas.lightning_module.MPflow`
- data module class: `hepattn.experiments.atlas.pflow_data.PflowDataModule`
- default `fit` config: `src/hepattn/experiments/atlas/configs/base.yaml`

Most behavior comes from:

- `src/hepattn/experiments/atlas/configs/base.yaml`
- `src/hepattn/experiments/atlas/pflow_data.py`
- `src/hepattn/experiments/atlas/lightning_module.py`
- `src/hepattn/models/wrapper.py`

Run commands from the ATLAS experiment directory unless you also update relative paths in the config:

```shell
cd src/hepattn/experiments/atlas
```

## Configuration

The default config is:

```text
configs/base.yaml
```

Important sections:

- `name`: run name. The CLI links this into the model name and logger experiment name.
- `seed_everything`: global training seed.
- `data`: ROOT file paths, feature/target settings, filtering limits, batch size, and scaling config.
- `trainer`: epochs, GPU settings, precision, output directory, logger, and callbacks.
- `model`: optimizer, LR schedule, MaskFormer architecture, matcher, and task losses.

Key data settings:

```yaml
data:
  train_path: /path/to/train.root
  valid_path: /path/to/val.root
  test_path: /path/to/test.root
  num_objects: 600
  max_nodes: 800
  num_train: -1
  num_val: -1
  num_test: -1
  batch_size: 128
  num_workers: 4
  incidence_cutval: 0.01
  scale_dict_path: configs/atlas_var_transform.yaml
```

`-1` means read all available entries for that split. Positive values read at most that many entries before dataset filtering.

The scaling config path is relative to the current working directory. If you run from `src/hepattn/experiments/atlas`, the default `configs/atlas_var_transform.yaml` resolves correctly.

## Data Files

The datamodule uses direct file paths:

- `data.train_path` for training
- `data.valid_path` for validation
- `data.test_path` for testing

Each configured path must point to one non-empty ROOT file. The dataset checks the path with `Path(path).is_file()` and then opens it with `uproot.open`.

The code expects each file to contain:

```text
EventTree
```

It does not scan a base directory, glob `*.root` files, or automatically merge multiple files. A symlink to a ROOT file should work, but a directory path will not. If a split is spread over many ROOT files, merge the files first or extend the datamodule to support a list of files.

## Data Flow

During `fit`, `PflowDataModule.setup("fit")` creates:

1. an `ATLASDataset` from `data.train_path`
2. an `ATLASDataset` from `data.valid_path`
3. train and validation dataloaders

During `test`, it creates one `ATLASDataset` from `data.test_path`.

The dataset flow is:

1. Open the ROOT file and read `EventTree`.
2. Load track, topocluster, truth particle, and association branches in chunks of 1000 events.
3. Build event counts for tracks, topoclusters, and particles.
4. Remove events with too many nodes or particles:

```text
n_tracks + n_topos >= max_nodes
n_particles >= num_objects
```

5. If `remove_wrong_idxs` is true, remove events where `len(track_particle_idx) != n_tracks`.
6. Flatten accepted events into tensors.
7. Normalize phi values and create sin/cos phi variables.
8. For each event, build padded node features from tracks followed by topoclusters.
9. Build padded truth particle targets.
10. Build the truth incidence matrix from track-particle and topo-particle associations.
11. Return `(inputs, labels)` to the dataloader.

The important model inputs include:

```text
node_features
node_valid
node_e
node_pt
node_eta
node_phi
node_sinphi
node_cosphi
node_is_track
```

The important labels include:

```text
particle_class
particle_valid
node_valid
particle_node_valid
particle_incidence
particle_e
particle_pt
particle_eta
particle_sinphi
particle_cosphi
event_number
mc_channel_number
```

`particle_node_valid` is computed from the incidence matrix using:

```text
particle_incidence > data.incidence_cutval
```

## Dry-Run Training

Use this before launching a real job. It disables the logger, uses one device, loads only a few events, and asks Lightning to run a minimal train/validation loop.

```shell
cd src/hepattn/experiments/atlas

CUDA_VISIBLE_DEVICES=5 python main.py fit \
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

Keep the options after `fit`; they are subcommand options. Use `CUDA_VISIBLE_DEVICES` to select the GPU for the quick check.

`fast_dev_run` is useful because it asks Lightning to run only a minimal train/validation loop while still checking config parsing, data loading, model forward, loss computation, and validation. `--trainer.logger=false` disables the logger, and `--trainer.callbacks=[]` clears configured callbacks so no callback expects logger or checkpoint state during the smoke test.

If you only want to disable logging without `fast_dev_run`, use:

```shell
--trainer.logger false
```

For small debug runs, these are usually the most useful knobs:

```shell
--data.num_train 10
--data.num_val 10
--data.batch_size 1
--data.num_workers 0
--trainer.devices 1
--trainer.max_epochs 1
--trainer.logger false
```

## Logger Configuration

The default ATLAS config uses:

```yaml
trainer:
  logger:
    class_path: hepattn.experiments.atlas.logger_utils.FixedCometLogger
    init_args:
      project: hepattn-atlas
```

With some installed Lightning/Comet versions, `project` is not accepted by `CometLogger`. If config parsing fails with an error like:

```text
Option 'project' is not accepted
Parser key "trainer.logger" does not validate
```

then either disable logging for the run:

```shell
--trainer.logger false
```

or update the config to use:

```yaml
trainer:
  logger:
    class_path: hepattn.experiments.atlas.logger_utils.FixedCometLogger
    init_args:
      project_name: hepattn-atlas
```

## Full Training

After the dry-run passes, launch training with the full config:

```shell
cd src/hepattn/experiments/atlas
python main.py fit -c configs/base.yaml
```

Common overrides:

```shell
python main.py fit \
  -c configs/base.yaml \
  --trainer.devices 1 \
  --trainer.precision bf16-mixed \
  --data.batch_size 64
```

Resume from a checkpoint:

```shell
python main.py fit \
  -c configs/base.yaml \
  --ckpt_path /path/to/checkpoint.ckpt
```

## Saved Outputs

In `fit` mode, the custom CLI rewrites `trainer.default_root_dir` into a timestamped run directory:

```text
<default_root_dir>/<name>_YYYYMMDD-THHMMSS/
```

For example, with:

```yaml
name: atlas_run4_jz1234_v1
trainer:
  default_root_dir: /home/lclissa/projects/hepattn/experiments/dryrun
```

the run directory looks like:

```text
/home/lclissa/projects/hepattn/experiments/dryrun/atlas_run4_jz1234_v1_YYYYMMDD-THHMMSS/
```

Local outputs include:

```text
<run_dir>/config.yaml
<run_dir>/metadata.yaml
<run_dir>/ckpts/*.ckpt
```

`metadata.yaml` is written by `hepattn.callbacks.SaveConfig`. Checkpoints are written by `hepattn.callbacks.Checkpoint` under `ckpts/`; filenames include the epoch and validation loss, for example:

```text
ckpts/epoch=003-val_loss=4.12345.ckpt
```

## Testing And Prediction Outputs

Testing uses the `test` subcommand:

```shell
python main.py test \
  -c /path/to/saved/config.yaml \
  --ckpt_path /path/to/checkpoint.ckpt \
  --data.test_path /path/to/test.root
```

For inference-style evaluation, set:

```shell
--data.is_inference true
```

or use the inference configs if appropriate:

```shell
python main.py test \
  -c configs/base_inference.yaml \
  -c configs/inference_override.yaml \
  --ckpt_path /path/to/checkpoint.ckpt
```

In `test` mode, `PflowPredictionWriter` saves predictions next to the checkpoint:

```text
<checkpoint_dir>/<checkpoint_name>__test.h5
<checkpoint_dir>/<checkpoint_name>__test.root
```

If `data.test_suff` is set, it is inserted into the filename.

When `--ckpt_path` is omitted in `test` mode, the CLI searches the config directory's `ckpts/` folder and picks the checkpoint with the lowest loss value parsed from the filename.
