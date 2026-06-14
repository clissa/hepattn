# CLIC Training

This document explains how `src/hepattn/experiments/clic/main.py` works in
`fit` mode, with emphasis on data loading, train/validation/test splits, input
features, important hyperparameters, and a small first-run recipe for new data.

## Entry Point

`src/hepattn/experiments/clic/main.py` is intentionally small. It starts a
Lightning CLI with:

- model class: `hepattn.experiments.clic.lightning_module.MPflow`
- data module class: `hepattn.experiments.clic.pflow_data.PflowDataModule`
- default `fit` config: `src/hepattn/experiments/clic/configs/base.yaml`

In other words, most fit-mode behavior comes from:

- `src/hepattn/experiments/clic/configs/base.yaml`
- `src/hepattn/experiments/clic/pflow_data.py`
- `src/hepattn/experiments/clic/lightning_module.py`
- `src/hepattn/models/wrapper.py`

## Data Files And Splits

Training, validation, and test data are not split internally from a single
file. The split is file-based and configured in `base.yaml`:

```yaml
data:
  train_path: /share/gpu1/syw24/dmitrii_clic/train_clic_fix.root
  valid_path: /share/gpu1/syw24/dmitrii_clic/val_clic_fix.root
  test_path: /share/gpu1/syw24/dmitrii_clic/test_clic_fix.root
```

During `fit`, `PflowDataModule.setup("fit")` creates:

- a training `CLICDataset` from `data.train_path`
- a validation `CLICDataset` from `data.valid_path`

`data.test_path` is only used in `test` mode.

The number of entries read from each file is controlled by:

```yaml
data:
  num_train: -1
  num_val: 5000
  num_test: -1
```

`-1` means all entries. A positive value means read at most that many ROOT tree
entries before dataset filtering. For example, `num_val: 5000` means "read up
to 5000 validation entries, then keep only the entries that pass the dataset
filters."

Each ROOT file must contain an `EventTree`. Data is read with `uproot`.

The configured paths must point to individual ROOT files. As written,
`CLICDataset` checks each path with `Path(path).is_file()` and then calls
`uproot.open(filepath)`. It does not scan a directory, glob `*.root` files, or
concatenate multiple files automatically. If a split is spread over many ROOT
files, create one merged file first or extend `PflowDataModule`/`CLICDataset`
to build a dataset over a list of files.

## Dataset Filtering

`CLICDataset` applies filtering after reading the requested number of entries.
By default, an event is removed if:

- `n_tracks + n_topos >= max_nodes`
- `n_particles >= num_objects`
- `remove_wrong_idxs` is true and `len(track_particle_idx) != n_tracks`

Important defaults:

```yaml
data:
  num_objects: 150
```

```python
CLICDataset(..., max_nodes=160, remove_wrong_idxs=True)
```

`max_nodes` and `remove_wrong_idxs` are constructor defaults in
`pflow_data.py`, not explicit defaults in `base.yaml`.

When the dataset is created, it prints how many events were removed for too
many nodes, too many particles, and mismatching `track_particle_idx`. These
messages are useful when checking new data.

## ROOT Branches Read

The dataset reads track, topocluster, particle, and auxiliary association
branches.

Track branches:

```text
track_pt
track_eta
track_phi
track_d0
track_z0
track_eta_int
track_phi_int
track_chi2
track_ndf
track_radiusofinnermosthit
track_tanlambda
track_omega
```

Topocluster branches:

```text
topo_eta
topo_phi
topo_rho
topo_e
topo_sigma_eta
topo_sigma_phi
topo_sigma_rho
topo_energy_ecal
topo_energy_hcal
topo_energy_other
```

Truth particle branches:

```text
particle_e
particle_pt
particle_eta
particle_phi
particle_pdgid
```

Auxiliary branches:

```text
particle_track_idx
track_particle_idx
topo2particle_topo_idx
topo2particle_particle_idx
topo2particle_energy
```

## Input Features

The model input is a padded sequence of nodes. Nodes are tracks followed by
topoclusters. The main model input tensor is:

```python
inputs["node_features"]
```

Its shape is approximately:

```text
(batch_size, max_nodes, 27)
```

The 27 node features are built in `CLICDataset.load_event`.

Common features:

```text
pt
eta
phi
cosphi
sinphi
```

Track interaction features:

```text
eta_int
phi_int
cosphi_int
sinphi_int
```

Track-only features, zero-filled for topoclusters:

```text
z0
d0
chi2
ndf
radiusofinnermosthit
tanlambda
omega
```

Topocluster-only features, zero-filled for tracks:

```text
e
rho
sigma_eta
sigma_phi
sigma_rho
energy_ecal
energy_hcal
energy_other
em_frac
```

Type flags:

```text
is_track
is_topo
```

The model also receives raw node variables:

```text
node_e
node_pt
node_eta
node_phi
node_sinphi
node_cosphi
node_is_track
```

These are listed under `model.model.init_args.raw_variables` in `base.yaml`.
They are used by later task heads, especially incidence and regression.

Feature scaling is configured by:

```yaml
data:
  scale_dict_path: configs/clic_var_transform.yaml
```

That path is relative to the current working directory. The documented command
in `src/hepattn/experiments/clic/README.md` runs from
`src/hepattn/experiments/clic`, where `configs/clic_var_transform.yaml` exists.

Note: in the current implementation, the constructed `node_features["cosphi"]`
and `node_features["sinphi"]` concatenate `topo_phi` for topocluster nodes,
while the raw features use `topo_cosphi` and `topo_sinphi`.

## Targets

The dataset returns `(inputs, labels)`.

Classification and validity labels:

```text
particle_class
particle_valid
node_valid
```

Mask/incidence labels:

```text
particle_node_valid
particle_incidence
```

Regression labels:

```text
particle_e
particle_pt
particle_eta
particle_sinphi
particle_cosphi
```

Event bookkeeping:

```text
event_number
```

The particle-to-node incidence matrix is built from:

```text
track_particle_idx
topo2particle_topo_idx
topo2particle_particle_idx
topo2particle_energy
```

Tracks get hard assignment weights of `1.0`. Topoclusters get energy
contribution weights from `topo2particle_energy`. Columns with no associated
particle are assigned to fake rows where possible. Each node column is then
normalized so its contributions sum to 1.

`data.incidence_cutval` controls the boolean mask target:

```yaml
data:
  incidence_cutval: 0.01
```

The label `particle_node_valid` is computed as:

```python
particle_incidence > incidence_cutval
```

## Model And Training Step

The configured model is `hepattn.models.MaskFormer`.

The `MPflow` Lightning module subclasses `ModelWrapper`. In each training step:

1. Unpack the batch into `inputs, targets`.
2. Run `outputs = self.model(inputs)`.
3. Compute task losses with `self.model.loss(outputs, targets)`.
4. Sum and log losses.
5. Periodically run prediction and metric logging every
   `trainer.log_every_n_steps`.
6. Return the total loss to Lightning.

Validation does the same forward and loss computation, then always runs
prediction and metric logging.

## Major Hyperparameters

Data:

```yaml
data:
  num_objects: 150
  num_workers: 16
  num_train: -1
  num_val: 5000
  num_test: -1
  batch_size: 512
  incidence_cutval: 0.01
```

Trainer:

```yaml
trainer:
  max_epochs: 200
  accelerator: gpu
  devices: 2
  precision: bf16-mixed
  gradient_clip_val: 0.1
  log_every_n_steps: 50
  default_root_dir: logs
```

Optimizer and LR schedule:

```yaml
model:
  optimizer: Lion
  lrs_config:
    initial: 1e-6
    max: 8e-5
    end: 1e-6
    pct_start: 0.05
    weight_decay: 1e-4
    skip_scheduler: false
```

The scheduler is `torch.optim.lr_scheduler.OneCycleLR`, configured in
`ModelWrapper.configure_optimizers`.

Architecture:

```yaml
model:
  model:
    init_args:
      dim: 256
      encoder:
        num_layers: 6
        attn_type: flash-varlen
        num_register_tokens: 8
        attn_kwargs:
          num_heads: 16
      decoder:
        num_decoder_layers: 4
        num_queries: 150
        mask_attention: true
```

Task heads and main loss weights:

```yaml
classification:
  object_ce: 2

mask:
  mask_bce: 5.0
  mask_dice: 1.0

incidence:
  kl_div: 1.0

regression:
  loss: l1
  loss_weight: 10.0
  cost_weight: 10.0
```

## First Run On New Data

For a smoke test, use a very small run before launching the full default
configuration:

```shell
cd src/hepattn/experiments/clic
python main.py fit \
  --config configs/base.yaml \
  --data.train_path /path/to/train.root \
  --data.valid_path /path/to/val.root \
  --data.num_train 100 \
  --data.num_val 50 \
  --data.batch_size 4 \
  --data.num_workers 0 \
  --trainer.devices 1 \
  --trainer.max_epochs 1 \
  --trainer.logger false \
  --trainer.callbacks=[]
```

Things to check first:

- The ROOT files exist and are non-empty.
- Each file contains an `EventTree`.
- All required branches listed above exist.
- The dataset printout says that at least some events remain after filtering.
- `node_features` has 27 features, matching `input_size: 27` in `base.yaml`.
- No event in the smoke-test sample exceeds `max_nodes` or `num_objects` unless
  you intentionally override those limits.
- If Comet is not configured, keep `--trainer.logger false` for the smoke test.
- If compilation or `flash-varlen` attention causes environment-specific
  issues, remove the compile callback and/or switch the attention type in the
  config.

For quick data compatibility debugging, the most useful knobs are:

```shell
--data.num_train 10
--data.num_val 10
--data.batch_size 1
--data.num_workers 0
--trainer.devices 1
--trainer.max_epochs 1
--trainer.logger false
```

## Saved Outputs

In `fit` mode, the CLI rewrites `trainer.default_root_dir` into a timestamped
run directory before classes are instantiated. With the default config:

```yaml
name: clic_v6
trainer:
  default_root_dir: logs
```

the run directory will look like:

```text
logs/clic_v6_YYYYMMDD-THHMMSS/
```

The path is resolved relative to the directory where the command is launched.
If you run from `src/hepattn/experiments/clic`, the outputs are under
`src/hepattn/experiments/clic/logs/...`. If you run from the repository root,
they are under `logs/...`.

The local fit outputs include:

```text
logs/clic_v6_YYYYMMDD-THHMMSS/config.yaml
logs/clic_v6_YYYYMMDD-THHMMSS/metadata.yaml
logs/clic_v6_YYYYMMDD-THHMMSS/ckpts/*.ckpt
```

`config.yaml` is the saved Lightning CLI configuration for the run.
`metadata.yaml` is written by `hepattn.callbacks.SaveConfig` and includes
dataset sizes, batch size, trainable parameter count, GPU info, package
versions, hostname, output directory, and logger URL when available.

Checkpoints are written by `hepattn.callbacks.Checkpoint` to the `ckpts/`
subdirectory. The filenames include epoch and monitored validation loss, for
example:

```text
ckpts/epoch=003-val_loss=4.12345.ckpt
```

The callback monitors:

```yaml
monitor: val/loss
```

and is configured with `save_top_k=-1` internally, so it saves every monitored
checkpoint rather than only the best one. Although `base.yaml` sets
`save_last: true`, this custom callback sets `self.save_last = False` during
setup.

Metrics are logged through the configured Lightning logger. The default config
uses `lightning.pytorch.loggers.CometLogger` with project `hepattn-clic`, so
train/validation losses, task metrics, hyperparameters, metadata, code assets,
and model checkpoints are sent to Comet when the logger is enabled and
configured. Local metric files are not explicitly written by this code path
unless the active Lightning logger does so.

In `test` mode, the CLIC `PflowPredictionWriter` writes prediction outputs next
to the checkpoint being evaluated:

```text
<checkpoint_dir>/<checkpoint_name>__test.h5
<checkpoint_dir>/<checkpoint_name>__test.root
```

If `data.test_suff` is set, it is inserted into the output filename.

## Evaluation Notes

The CLIC README recommends special flags for real performance evaluation in
`test` mode:

```shell
python main.py test \
  -c <path to config.yaml> \
  --data.test_path test_clic_common_infer.root \
  --data.is_inference true \
  --trainer.precision 32-true \
  --matmul_precision highest
```

That is separate from `fit` mode, but relevant when moving from a smoke test to
real evaluation. The README also notes that evaluation may require changing the
attention type to `torch` and removing the compile callback.
