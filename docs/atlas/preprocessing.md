# ATLAS Preprocessing

This note summarizes the ATLAS particle-flow data path used by the current
`src/hepattn/experiments/atlas` workflow. It is based on:

- `src/hepattn/experiments/atlas/main.py`
- `src/hepattn/experiments/atlas/configs/base.yaml`
- `src/hepattn/experiments/atlas/configs/atlas_var_transform.yaml`
- `src/hepattn/experiments/atlas/pflow_data.py`
- `src/hepattn/models/input.py`
- `src/hepattn/models/maskformer.py`

## Entry Point

ATLAS training starts from:

```shell
cd src/hepattn/experiments/atlas
python main.py fit -c configs/base.yaml
```

`main.py` wires the custom Lightning CLI to:

- model class: `hepattn.experiments.atlas.lightning_module.MPflow`
- data module: `hepattn.experiments.atlas.pflow_data.PflowDataModule`
- default fit config: `configs/base.yaml`

## Data Source

The datamodule reads one ROOT file per split:

- `data.train_path`
- `data.valid_path`
- `data.test_path`

Each path must point to a non-empty file. The dataset checks this with
`Path(path).is_file()` and opens the file with `uproot.open`. The expected tree
name is:

```text
EventTree
```

The code does not scan directories, glob multiple ROOT files, or merge files.
If a split is distributed over multiple ROOT files, the inputs need to be
merged first or the datamodule needs to be extended.

`data.num_train`, `data.num_val`, and `data.num_test` control how many entries
are considered from each split before event filtering. A value of `-1` means
use all entries in the file.

## ROOT Branches Read

`ATLASDataset.init_variables_list()` defines the branches loaded from the ROOT
tree.

Track branches:

```text
track_pt
track_eta
track_phi
track_d0
track_z0
track_eta_int
track_phi_int
```

Topocluster branches:

```text
topo_e
topo_pt
topo_eta
topo_phi
topo_num_cells
topo_em_frac
topo_center_mag
topo_center_lambda
```

Truth-particle branches:

```text
particle_e
particle_pt
particle_eta
particle_phi
particle_pdgid
```

Association and metadata branches:

```text
particle_track_idx
track_particle_idx
topo2particle_topo_idx
topo2particle_particle_idx
topo2particle_energy
eventNumber
mcChannelNumber
```

The loader reads these branches in chunks of 1000 events to avoid large jagged
array overflows in uproot.

## Event Selection

After loading event counts, the dataset removes events that exceed the
configured object limits:

```text
n_tracks + n_topos >= data.max_nodes
n_particles >= data.num_objects
```

With the current base config these limits are:

```yaml
data:
  max_nodes: 800
  num_objects: 600
```

If `remove_wrong_idxs` is true, the dataset also removes events where:

```text
len(track_particle_idx) != n_tracks
```

`remove_wrong_idxs` is an `ATLASDataset` constructor argument accepted through
the datamodule `**kwargs`; it is not explicitly set in the current base config,
so the dataset default is used.

## Runtime Preprocessing

The ATLAS dataset does preprocessing at training/runtime rather than through a
separate `prep.py`.

The main transformations are:

- flatten accepted jagged event arrays into contiguous tensors
- normalize `track_phi`, `track_phi_int`, `topo_phi`, and `particle_phi` with
  `atan2(sin(phi), cos(phi))`
- create `sinphi` and `cosphi` for non-particle phi fields
- map `particle_pdgid` to internal classes
- build one padded node sequence per event, with tracks first and
  topoclusters second
- build padded truth-particle targets
- build particle-node incidence targets from track-particle and
  topo-particle association branches
- create fake particle rows for nodes with no associated particle, as long as
  there is room below `num_objects`
- normalize each node column of the incidence matrix so node contributions sum
  to one

For non-inference training, particles without associated tracks are remapped:

- trackless charged hadrons and electrons are shifted to neutral classes
- trackless muons become neutral hadrons

When `data.is_inference: true`, this trackless-particle class remapping is
disabled.

## Scaling

Scaling is configured by:

```yaml
data:
  scale_dict_path: configs/atlas_var_transform.yaml
```

`FeatureScaler` applies a transform only for keys present in that YAML file.
Missing keys use an identity transform.

The current scaling file defines transforms for:

```text
eta
pt
e
d0
z0
num_cells
center_mag
center_lambda
```

`pt`, `e`, and `center_lambda` use a square-root transform before min-max-style
scaling. `eta`, `d0`, `z0`, `num_cells`, and `center_mag` use symmetric
min-max-style scaling. `em_frac`, phi variables, sin/cos variables, and type
flags are not explicitly listed in the scaling YAML, so they are passed through
unchanged.

Particle regression labels for `e`, `pt`, and `eta` are also transformed with
the same scaler before being returned as targets. `sinphi` and `cosphi` are not
scaled.

## Batch Inputs

Each dataset item returns `(inputs, labels)`. The ATLAS model-facing input
dictionary contains:

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

`node_valid` marks real tracks/topoclusters inside the padded node sequence.
The other `node_*` variables after `node_features` are raw or lightly derived
per-node quantities used as `raw_variables` by the model, especially by the
incidence-based regression task.

## Labels

The label dictionary contains:

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
getitem_idx
```

`particle_node_valid` is a boolean mask derived from the normalized incidence
matrix:

```text
particle_incidence > data.incidence_cutval
```

The current base config sets:

```yaml
data:
  incidence_cutval: 0.01
```

## Features Passed To The Model

The current model config has one `InputNet`:

```yaml
input_name: node
fields:
  - features
net:
  input_size: 18
posenc:
  fields: [eta, phi]
```

This means `InputNet` looks up `inputs["node_features"]`. The model sees this
tensor by name, but the tensor itself is a packed feature matrix with a
position-based column convention.

The packed `node_features` tensor has shape:

```text
[max_nodes, 18] per event
```

After normal collation by the PyTorch dataloader, the batched shape is:

```text
[batch_size, max_nodes, 18]
```

The current 18 columns, in order, are:

| Index | Feature | Tracks | Topoclusters |
| ---: | --- | --- | --- |
| 0 | `pt` | scaled `track_pt` | scaled `topo_pt` |
| 1 | `eta` | scaled `track_eta` | scaled `topo_eta` |
| 2 | `phi` | `track_phi` | `topo_phi` |
| 3 | `cosphi` | `cos(track_phi)` | `cos(topo_phi)` |
| 4 | `sinphi` | `sin(track_phi)` | `sin(topo_phi)` |
| 5 | `eta_int` | scaled `track_eta_int` | `0` |
| 6 | `phi_int` | `track_phi_int` | `0` |
| 7 | `cosphi_int` | `cos(track_phi_int)` | `0` |
| 8 | `sinphi_int` | `sin(track_phi_int)` | `0` |
| 9 | `z0` | scaled `track_z0` | `0` |
| 10 | `d0` | scaled `track_d0` | `0` |
| 11 | `e` | `0` | scaled `topo_e` |
| 12 | `num_cells` | `0` | scaled `topo_num_cells` |
| 13 | `em_frac` | `0` | `topo_em_frac` |
| 14 | `center_mag` | `0` | scaled `topo_center_mag` |
| 15 | `center_lambda` | `0` | scaled `topo_center_lambda` |
| 16 | `is_track` | `1` | `0` |
| 17 | `is_topo` | `0` | `1` |

The node sequence itself is also position-based:

```text
[all tracks, then all topoclusters, then zero padding up to max_nodes]
```

`InputNet` embeds the 18 packed columns with a dense network to the model
dimension. The positional encoder separately reads named fields:

```text
node_eta
node_phi
```

These positional-encoding fields come from the raw node features, not from the
packed/scaled `node_features` columns.

The model config also lists these `raw_variables`:

```text
node_e
node_pt
node_eta
node_sinphi
node_cosphi
node_is_track
```

`MaskFormer` copies those tensors into its internal state if present. They are
not embedded by the `InputNet`, but they are available to downstream tasks.
In the current config, the incidence-based regression task uses node-level
information in addition to query embeddings and proxy features.

## Configuration Caveat

`configs/base.yaml` still has:

```yaml
data:
  inputs:
    hit:
      - x
      - y
      - z
      - r
      - s
      - theta
      - phi
```

For the current ATLAS dataset and model, this does not describe the actual
model input. `ATLASDataset.__getitem__` returns `node_*` keys, and the model
embeds `node.features`. Treat `node_features` and the `InputNet` config as the
authoritative model-facing feature definition for this workflow.

## Verification Notes

This document was produced by reading the local source and configs. It was not
validated by loading external ATLAS data or running a training job. Data loading
tests for this path would require suitable ROOT files with the expected
`EventTree` and branches.
