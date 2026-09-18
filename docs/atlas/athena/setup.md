# Run-4 GLOW / GLOW-MDN architecture handoff

These files are notes and executable examples for Athena developers to import
or copy the existing model architecture. Athena integration and export are
left to the receiving developers. The checkpoints are supplied separately.

Keep these files together in any directory:

| File | Purpose |
| --- | --- |
| `atlas-glow-run4.py` | Construct either architecture, load weights, invoke inference, decode kinematics |
| `glow.yaml` | Resolved model definition for GLOW epoch 79 |
| `glow-mdn.yaml` | Resolved model definition for GLOW-MDN epoch 76 |
| `atlas_var_transform.yaml` | Shared input and regression scaling constants |
| `setup.md` | This guide |

The Python file imports `hepattn`; it does not contain a duplicate standalone
implementation. All repository paths below are relative to the repository root,
not to the directory containing this guide. Repository URL:

- [clissa/hepattn](https://github.com/clissa/hepattn) (`main` branch is up-to-date)
- [ATLAS GitLab -- GLOW](https://gitlab.cern.ch/atlas-jetetmiss/pflow/commontools/glow#) (check `main` branch, push ongoing).

## Exact checkpoints

Under `results/long-run/`, the selected pairs are:

```text
GLOW_jz1234_v1-full/
  atlas_GLOW_jz1234_v1-full_20260831-T140852/
    config.yaml
    ckpts/epoch=079-val_loss=5.21820.ckpt

MDN_jz1234_v1-full/
  atlas_MDN_jz1234_v1-full_20260819-T212546/
    config.yaml
    ckpts/epoch=076-val_loss=3.18772.ckpt
```

## Quick-start

This runs both checkpoints on a small synthetic event. You need the five files
listed above, the two checkpoints, and a checkout or installation of `hepattn`.
No dataset or Athena installation is needed for this check.

1. **Activate the project environment.** Use Python 3.12 with the repository's
   dependencies installed, following its `README.md` and `pyproject.toml` if
   you have not set up an environment yet. FlashAttention must be importable
   even for this CPU check because the current model source imports it
   unconditionally. For an existing virtual environment:

   ```bash
   source /path/to/your/environment/bin/activate
   ```

2. **Set the paths and check imports.** Replace the example paths below with
   your checkout, handoff directory, and checkpoint locations. The checkpoint
   files can be stored anywhere; their original locations are listed above.

   ```bash
   export HEPATTN_REPO=/path/to/hepattn
   export PYTHONPATH="$HEPATTN_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
   cd /path/to/shared-handoff-files

   GLOW_CHECKPOINT='/path/to/epoch=079-val_loss=5.21820.ckpt'
   MDN_CHECKPOINT='/path/to/epoch=076-val_loss=3.18772.ckpt'

   python -c 'import torch, yaml, jsonargparse; from hepattn.models import MaskFormer; print("Model imports OK; PyTorch", torch.__version__)'
   ```

   If `hepattn` is already installed in the active environment, the two
   `export` lines are unnecessary. Resolve any import errors before continuing.

3. **Load and run both models.** These commands use CPU float32 inference with
   PyTorch attention and save the resulting tensors for inspection:

   ```bash
   python atlas-glow-run4.py --config glow.yaml \
     --checkpoint "$GLOW_CHECKPOINT" --smoke-test --output glow-smoke.pt

   python atlas-glow-run4.py --config glow-mdn.yaml \
     --checkpoint "$MDN_CHECKPOINT" --smoke-test --output glow-mdn-smoke.pt
   ```

   Both commands should exit without errors and print output names, shapes,
   and dtypes. For the synthetic event, look for
   `classification.pflow_class_prob: (1, 600, 6)` and
   `regression.pflow_regr: (1, 600, 5)` in both runs. The MDN run additionally
   prints `regression.pflow_mdn_means: (1, 600, 1, 2)` and matching scale
   dimensions. The files `glow-smoke.pt` and `glow-mdn-smoke.pt` each contain
   raw `final` outputs and decoded `particles` with a non-null `valid` mask.

A successful run checks strict checkpoint loading and a forward pass with
finite raw outputs. It does not establish physics agreement on real events.
These commands still need execution in a working project environment; see
[Validation status](#validation-status) for the checks performed for this
handoff. Continue below for architecture details, GPU invocation, real-input
tensor conventions, and output interpretation.

## Configuration provenance

Each supplied architecture YAML is extracted from that run's resolved
`model.model` section. Only the training matcher is set to `null` and the scaler
path is changed to the adjacent `atlas_var_transform.yaml`. Training paths,
logging credentials/settings, and data paths are not needed. Constructor loss
settings are retained to preserve the original task configuration and buffers;
inference does not compute losses or matching.

The scaler is copied from
`src/hepattn/experiments/atlas/configs/atlas_var_transform.yaml` in the inspected
checkout. It is a separate dependency: weights do not encode these transforms.
This guide describes source at commit
`6dc3eb01166407ab640e86f855536f9e22eae3d4`, not a claim that this is the training
commit or that an original training-environment replay has been verified.

## Where the model is defined

The direct import is:

```python
from hepattn.models import MaskFormer
# Equivalent: from hepattn.models.maskformer import MaskFormer
```

`MaskFormer` is an ordinary `torch.nn.Module`. Its constructor takes the input
network, encoder, decoder, and prediction tasks. `build_model()` in the supplied
script instantiates those objects from the selected YAML using `jsonargparse`,
the same configuration mechanism used by the project.

| Repository module | What to import or copy |
| --- | --- |
| `src/hepattn/models/maskformer.py` | `MaskFormer`: shared assembly, forward pass, learned queries, task invocation |
| `src/hepattn/models/input.py` | `InputNet`: pack/embed the node features |
| `src/hepattn/models/posenc.py` | `FourierPositionEncoder`: raw eta/phi encoding |
| `src/hepattn/models/transformer.py` | `Encoder`, encoder layers, residual and normalization arrangement |
| `src/hepattn/models/decoder.py` | `MaskFormerDecoder` and its layers, including mask-controlled attention |
| `src/hepattn/models/attention.py` | Attention projections, backends, padding-mask convention |
| `src/hepattn/models/task.py` | Classification, constituent mask, incidence, and the two regression heads |
| `src/hepattn/models/dense.py`, `norm.py`, `activation.py` | Supporting network blocks |
| `src/hepattn/utils/scaling.py` | `FeatureScaler`, `VarTransform`: exact transforms and inverses |
| `src/hepattn/experiments/atlas/pflow_data.py` | `ATLASDataset.load_event()` and `__getitem__()`: input feature construction |

The deterministic head is `IncidenceBasedRegressionTask`; the MDN head is
`IncidenceBasedMixtureRegressionTask`, which shares its proxy construction.
Copying only the head is insufficient to recreate the model. If porting the
code, follow its imports into the supporting modules and preserve the parameter
names for checkpoint loading. The `MPflow` class in
`src/hepattn/experiments/atlas/lightning_module.py` is the training wrapper;
developers do not need a Lightning trainer to invoke `MaskFormer`.

Both selected models have 18 input features, a 256-dimensional embedding,
6 encoder layers with 16 heads and 8 register tokens, 600 learned particle
queries, and 4 masked decoder layers with 16 heads. The configured regression
network takes 518 values per query: 256 query features, 5 proxy kinematics,
1 charged flag, and 256 incidence-weighted node features. Its hidden widths are
512, 256, 128, 64, 32. GLOW ends in 5 outputs; GLOW-MDN ends in 8.

## Environment and invocation

Use a Python 3.12 environment with the repository dependencies available; the
authoritative environment definition is its `pyproject.toml` and setup guidance
is in its `README.md`. The saved run metadata records PyTorch `2.7.0+cu126`,
CUDA `12.6`, and Lightning `2.5.0.post0`. The project specifies
`jsonargparse[signatures]` and FlashAttention `2.7.4.post1`; PyYAML is also used
by this script and the scaler. Keep PyTorch/CUDA/FlashAttention builds compatible
with the project's environment definition.

For developers who already have that environment, make the source importable
without running any training entrypoint:

```bash
# Replace this with the path to the hepattn checkout supplied to you.
export HEPATTN_REPO=/path/to/hepattn
export PYTHONPATH="$HEPATTN_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
cd /path/to/shared-handoff-files
```

An installed `hepattn` package is also sufficient; in that case `PYTHONPATH` is
unnecessary. The current source imports `flash_attn` unconditionally from
`attention.py`, even when using the PyTorch backend. Thus a torch-only
environment without an importable FlashAttention installation is not sufficient
for this import-based handoff. Developers copying code can adapt the backend
imports in their integration.

Run a synthetic smoke check for each already-available checkpoint:

```bash
python atlas-glow-run4.py --config glow.yaml \
  --checkpoint '/path/to/epoch=079-val_loss=5.21820.ckpt' --smoke-test

python atlas-glow-run4.py --config glow-mdn.yaml \
  --checkpoint '/path/to/epoch=076-val_loss=3.18772.ckpt' --smoke-test
```

The default is float32 inference on CPU with PyTorch SDPA (`attn_type: torch`).
The saved encoder configuration uses `flash-varlen`; the script changes only
that backend at construction, retaining parameter names and shapes. To use the
original encoder backend, add `--device cuda --attention flash-varlen`; the CLI
then uses bfloat16 autocast. SDPA/float32 and FlashAttention/bfloat16 are not
expected to be bitwise identical. Compare them on representative events before
choosing the production backend.

The synthetic event has two tracks, two topoclusters, and four padding slots.
It checks loading and tensor plumbing, not reconstruction quality. For actual
model-ready input tensors saved with `torch.save(inputs, "event.pt")`, replace
`--smoke-test` with `--inputs event.pt --output prediction.pt`. Input files
contain only the dictionary described below. The output file contains `final`
(nested raw final task outputs) and `particles` (decoded quantities, still in
600 query slots). No ROOT input, truth labels, optimizer, or trainer is required.

To import the supplied file despite its hyphenated name:

```python
import importlib.util
from pathlib import Path
import torch

handoff = Path("/path/to/shared-handoff-files")
spec = importlib.util.spec_from_file_location("atlas_glow_run4", handoff / "atlas-glow-run4.py")
glow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(glow)

model = glow.load_model(handoff / "glow-mdn.yaml", "/path/to/checkpoint.ckpt")
inputs = glow.example_inputs(model)  # Replace with your event adapter's tensors.
with torch.inference_mode():
    outputs = model(inputs)
    final = outputs["final"]
    predictions = model.predict({"final": final})["final"]
    particles = glow.physical_particles(model, final)

# First event's non-null slots, in physical units:
selected_pt = particles["pt"][0, particles["valid"][0]]
```

`load_model()` extracts `checkpoint["state_dict"]` entries beginning with
`model.`, removes that wrapper prefix, and calls `load_state_dict(strict=True)`.
Other wrapper state is outside this inference model. Do not use `strict=False`
to hide an architecture/checkpoint mismatch. Both `tasks.*` and
`decoder.tasks.*` entries are expected because the same task modules are
registered in both places. `build_model()` alone constructs untrained weights;
use `load_model()` for inference.

## Input tensor contract

Let `B` be the batch size, `N` the padded number of detector nodes, and `Q=600`
the fixed number of output queries. All tensors are on the same device. Use
float32 inputs except for the boolean validity mask; the CLI handles autocast.

| Key | Shape | Meaning |
| --- | --- | --- |
| `node_features` | `[B, N, 18]` | Packed/scaled features in the table below |
| `node_valid` | `[B, N]` | Boolean: `True` means real node; `False` means padding |
| `node_e` | `[B, N]` | Raw topocluster energy; zero for tracks |
| `node_pt` | `[B, N]` | Raw track transverse momentum; **zero for topoclusters** |
| `node_eta` | `[B, N]` | Raw track/topocluster eta, also used for positional encoding |
| `node_phi` | `[B, N]` | Raw track/topocluster phi in radians, also used for positional encoding |
| `node_sinphi`, `node_cosphi` | Each `[B, N]` | Sine/cosine of the raw node phi |
| `node_is_track` | `[B, N]` | Float flag: tracks 1, topoclusters/padding 0 |

The sequence is tracks first, topoclusters second, then zero padding. All
features, including the raw auxiliary tensors and both flags, are zero in
padding slots. Missing type-specific features are filled with **zero after
scaling**, not with the transform of zero. Supply finite inputs and at least
one real node per event; empty-event behavior has not been validated.

The saved data configs use `N=800` and `Q=600`. The model accepts a different
`N`, as in the smoke example; increasing multiplicity is not a validated change
of physics coverage. The training loader keeps events with fewer than 800 nodes
and fewer than 600 truth particles, rather than truncating them. An Athena
adapter must decide how to handle events outside that scope.

In this table `S_name(x)` means the supplied scaler's transform for `name`:

| Column | Feature | Track value | Topocluster value |
| --- | --- | --- | --- |
| 0 | pt | `S_pt(track_pt)` | `S_pt(topo_pt)` |
| 1 | eta | `S_eta(track_eta)` | `S_eta(topo_eta)` |
| 2 | phi | `track_phi` | `topo_phi` |
| 3 | cosphi | `cos(track_phi)` | `cos(topo_phi)` |
| 4 | sinphi | `sin(track_phi)` | `sin(topo_phi)` |
| 5 | eta_int | `S_eta(track_eta_int)` | 0 |
| 6 | phi_int | `track_phi_int` | 0 |
| 7 | cosphi_int | `cos(track_phi_int)` | 0 |
| 8 | sinphi_int | `sin(track_phi_int)` | 0 |
| 9 | z0 | `S_z0(track_z0)` | 0 |
| 10 | d0 | `S_d0(track_d0)` | 0 |
| 11 | e | 0 | `S_e(topo_e)` |
| 12 | num_cells | 0 | `S_num_cells(topo_num_cells)` |
| 13 | em_frac | 0 | `topo_em_frac` |
| 14 | center_mag | 0 | `S_center_mag(topo_center_mag)` |
| 15 | center_lambda | 0 | `S_center_lambda(topo_center_lambda)` |
| 16 | is_track | 1 | 0 |
| 17 | is_topo | 0 | 1 |

`eta_int`/`phi_int` are the supplied track extrapolation coordinates at the
calorimeter; `d0`/`z0` are track impact parameters. The topo features describe
energy, cell count, electromagnetic fraction, and cluster geometry/shower
depth. Match the upstream `EventTree` branch definitions when choosing Athena
accessors: this repository consumes those quantities and does not define the
extrapolation surface or upstream moment calculation.

The analysis code labels energy and pt in GeV. Convert Athena's energy units
to the training branch convention before scaling. Angular inputs are radians
and eta is dimensionless. Preserve the upstream length/moment units for `d0`,
`z0`, `center_mag`, and `center_lambda`; the loader contains no conversion or
unit metadata for those branches, so their precise Athena accessor/unit mapping
needs confirmation against the ntuple producer. Do not infer those mappings
from the scaler ranges alone. The loader wraps phi using
`atan2(sin(phi), cos(phi))`.

Each configured transform is `(f(x) - shift) / scale`, where
`shift=(max+min)/2`, `scale=(max-min)/2`, and `f` is square root for `e`, `pt`,
and `center_lambda`, otherwise identity. The `min`/`max` values apply **after**
the square root. There is no clipping. The YAML's `mean`/`std` values are not
used for these `min_max_sym` transforms. Fields absent from the YAML, including
`em_frac`, `sinphi`, and `cosphi`, use identity transforms. Use
`FeatureScaler` rather than a conventional fitted standardization transform.

## Outputs and physical interpretation

`model(inputs)` returns intermediate decoder outputs plus `outputs["final"]`.
Use the final dictionary for reconstructed particles. Queries are an unordered
set of candidate slots; their indices are not truth-particle or input-node IDs.
The input-node axis of the association tensors does retain input ordering.

| Final task/key | Shape | Meaning |
| --- | --- | --- |
| `classification.pflow_class_prob` | `[B, Q, 6]` | **Logits**, despite the name; softmax over the last axis gives probabilities |
| `mask.pflow_node_logit` | `[B, Q, N]` | Constituent-association logits; sigmoid gives association scores |
| `incidence.pflow_incidence` | `[B, Q, N]` | Fractional node assignments, softmax over **queries**; sums to one over Q for real nodes and zero for padding |
| `regression.pflow_regr` | `[B, Q, 5]` | Scaled point estimates ordered **e, pt, eta, sinphi, cosphi** |
| `regression.pflow_proxy_regr` | `[B, Q, 5]` | Scaled charged/neutral proxy used by the regression |
| `regression.pflow_proxy_ch_regr` | `[B, Q, 5]` | Charged proxy, zeroed for non-charged predictions |
| `regression.pflow_proxy_neut_regr` | `[B, Q, 5]` | Neutral proxy before choosing the charged/neutral branch |
| `regression.pflow_is_charged` | `[B, Q]` | Predicted class is 0, 1, or 2 |

Class indices are 0 charged hadron, 1 electron, 2 muon, 3 neutral hadron,
4 photon, 5 null/residual/fake. These are categories, not signed PDG IDs or
electric-charge estimates. The dataset relabels trackless charged hadrons to
neutral hadrons, trackless electrons to photons, and trackless muons to neutral
hadrons. Inference selects non-null slots with `argmax(logits) < 5`.

`model.predict()` applies argmax classification, a sigmoid mask threshold of
`>= 0.1`, and splits the regression vector into named fields. **It does not
inverse-transform the regression.** The supplied `physical_particles()` does
that step and computes `phi=atan2(sinphi, cosphi)`. It retains every slot and
returns a `valid` mask. Additional analysis or Athena selection cuts are not
applied. No constraint is imposed that predicted E equals pt*cosh(eta), or that
the two predicted angular components lie exactly on the unit circle.

The mask and incidence predictions have different meanings. The mask controls
decoder attention and predicts constituent membership; incidence estimates
fractional assignments used in the kinematic proxy. The charged proxy uses the
source's incidence-based track selection and sets E=pt*cosh(eta). The neutral
proxy combines calorimeter energy and energy-weighted direction. Both variants
then predict corrections using the proxy, query, and weighted node embeddings.
Preserve `get_proxy_feats()` when porting: its track selection includes
per-track query competition and recovery logic, not just an independent argmax
over tracks for each query.

### Extra GLOW-MDN outputs

This checkpoint uses **K=1**, a single diagonal Gaussian component over the
two scaled variables `[e, pt]`. Directions are deterministic.

| Regression key | Shape | Meaning |
| --- | --- | --- |
| `pflow_mdn_log_weights` | `[B, Q, 1]` | Log mixture weights (zero for K=1) |
| `pflow_mdn_means` | `[B, Q, 1, 2]` | Means of scaled e and pt, including proxy offsets |
| `pflow_mdn_scales` | `[B, Q, 1, 2]` | Positive standard deviations in scaled space |
| `pflow_deterministic_regr` | `[B, Q, 3]` | Scaled eta, sinphi, cosphi including proxy offsets |

The eight raw head values are one mixture logit, two mean corrections, two
unconstrained scale values, and three deterministic corrections. Scales are
`softplus(raw_scale + scale_offset) + 0.001`, with `scale_offset` stored in the
checkpoint. `pflow_regr` combines the mixture-weighted scaled means and the
deterministic direction. Inverse-transforming that point reproduces the model's
point-output convention; because the inverse E/pt transform is quadratic, it
is not the expectation of E/pt in physical space. Scales are not GeV errors,
and must not be inverse-transformed as though they were means. Retain these raw
outputs if Athena needs distribution information: the inherited `predict()`
returns point/proxy fields and does not propagate the MDN parameters.

## Validation status

The architecture YAMLs were checked against the two saved resolved configs,
allowing only the matcher and scaler-path changes described above. The shared
scaler was checked byte-for-byte against the repository file. Checkpoint
metadata inspection confirmed epochs 79/76, query weights of shape `[600,256]`,
and final regression weights of shape `[5,32]` / `[8,32]`, respectively.

Python syntax, Ruff lint, and Ruff formatting checks were run on the handoff
script. Full checkpoint deserialization into PyTorch, strict state loading,
forward inference, and CPU/GPU numerical comparisons were **not run** for this
handoff. In the development checkout, enter the PyTorch container with
`source ~/start_dev/start_container_pytorch.sh`, then run
`source ~/projects/hepattn/.hepattn/bin/activate` inside that container shell.
The virtual environment depends on the container; host-only interpreter errors
do not establish that it is broken. The smoke commands above must be run in
that activated environment before treating this as runtime-validated.
Athena event preprocessing and representative-event physics agreement remain
integration validation tasks.
