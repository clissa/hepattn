# ATLAS deterministic regression baseline

This document freezes the ATLAS incidence-based regression behavior before the
v1 mixture-density-network (MDN) implementation. It records repository truth at
commit `da262d75f6f73d05b607183b13f4334d8e07bd6c`; it is not a claim of exact
training reproducibility or a tuned scientific result.

## Experiment scope

The reference experiment is
`src/hepattn/experiments/atlas/configs/base_zj1234.yaml`.

- Run name: `atlas_jz1234_v0_nopart_reproduce`.
- Output directory:
  `/home/lclissa/projects/hepattn/results/jz1234_v0_nopart_reproduce`.
- Global Lightning seed: `20260704`.
- Training budget: 50 epochs, four GPUs, `bf16-mixed` precision, batch size
  112, and all available train/validation events (`num_train: -1`,
  `num_val: -1`).
- The configured train, validation, and test ROOT files are separate JZ1234
  partitions under the experiment author's local `/fast_scratch_4` tree.
  These paths are environment-specific and the data are not part of this
  repository.
- Events are padded to at most 600 truth particles and 800 detector nodes.

The configured seed is not the workflow's only random-state input. As described
in `docs/atlas/seeding.md`, the ATLAS dataset also has fixed dataset-level seed
behavior. Reproduction therefore also depends on the resolved config, data
files, software stack, hardware, and checkpoint; GPU execution is not assumed
to be bitwise deterministic.

## Deterministic incidence-based task

The regression task is
`hepattn.models.task.IncidenceBasedRegressionTask` with these baseline settings:

```yaml
fields: [e, pt, eta, sinphi, cosphi]
loss: l1
loss_weight: 10.0
cost_weight: 10.0
use_incidence: true
use_nodes: true
cost: new
mode: offset
has_intermediate_loss: false
```

Its dense head receives 518 features and directly produces five outputs in the
same order as `fields`. The task runs only on the final decoder output.

The task input concatenates the query embedding, a five-field proxy, and a
charged indicator. With `use_nodes: true`, it also appends the incidence-weighted
node embeddings. Predicted incidence values and class probabilities are detached
before proxy construction. In offset mode, the five dense-head values are added
to the corresponding proxy values.

## Proxy construction

Proxy construction uses the predicted incidence matrix, predicted class, and
raw node features.

- A query is classified as charged when the predicted class index is below
  three.
- For charged queries, the implementation selects one associated track from
  the incidence scores. Its track kinematics form the proxy; energy is replaced
  by `pt * cosh(eta)`.
- For neutral queries, calorimeter-node energy weights are formed from
  incidence times node energy after tracks are excluded. The neutral proxy uses
  the weighted calorimeter direction, summed energy, and
  `pt = e / cosh(eta)`.
- The charged proxy is retained only for charged queries. The neutral proxy is
  retained only for neutral queries. Separate charged and neutral proxy tensors
  are also returned for diagnostics.

The five proxy fields are transformed with the same `FeatureScaler` used by the
dataset targets before they are added to the network offsets.

## Feature representation and scaling

Targets and proxies are represented as `[e, pt, eta, sinphi, cosphi]`.
`src/hepattn/experiments/atlas/configs/atlas_var_transform.yaml` defines:

- `e`: square root followed by symmetric min-max scaling over `[0.001, 20]`.
- `pt`: square root followed by symmetric min-max scaling over `[0.001, 20]`.
- `eta`: symmetric min-max scaling over `[-3, 3]`.
- `sinphi` and `cosphi`: no explicit transform, so the scaler's identity
  transform is used.

Consequently the regression loss and matching cost operate in this scaled
five-dimensional space. In particular, an average or offset in scaled
square-root `e`/`pt` space is not generally an average or offset in physical
energy or transverse momentum after the nonlinear inverse transform.

## Loss and Hungarian matching

The deterministic task stacks the five scaled truth fields and computes
elementwise L1 error against the five point predictions. Only entries selected
by `particle_valid` contribute. The errors are reduced with a mean over all
valid objects and all five fields, then multiplied by `loss_weight: 10.0`.

For matching, `cost: new` detaches the point predictions, casts predictions and
targets to float32, computes the mean pairwise five-field L1 error, and
multiplies it by `cost_weight: 10.0`. The resulting axes are batch, predicted
query, and truth particle. This regression cost is added to the other enabled
task costs before the configured SciPy Hungarian solver runs.

`MaskFormer.loss()` uses the matching indices to permute every tensor named in
each task's `outputs` list. For this regression task that list contains the point
regression, combined proxy, charged proxy, neutral proxy, and charged-indicator
tensors. Losses are evaluated only after this permutation.

## Prediction interface

`IncidenceBasedRegressionTask.predict()` exposes the following scaled tensors:

```text
pflow_e, pflow_pt, pflow_eta, pflow_sinphi, pflow_cosphi
pflow_proxy_e, pflow_proxy_pt, pflow_proxy_eta,
pflow_proxy_sinphi, pflow_proxy_cosphi
pflow_proxy_ch_e, pflow_proxy_ch_pt, pflow_proxy_ch_eta,
pflow_proxy_ch_sinphi, pflow_proxy_ch_cosphi
pflow_proxy_neut_e, pflow_proxy_neut_pt, pflow_proxy_neut_eta,
pflow_proxy_neut_sinphi, pflow_proxy_neut_cosphi
pflow_is_charged
```

During test inference, `PflowPredictionWriter` inverse-transforms the five point,
combined-proxy, charged-proxy, and neutral-proxy fields and stores them in the
HDF5 `regression` dataset as `pred_*`, `proxy_*`, `proxy_ch_*`, and
`proxy_neut_*`, alongside `truth_*` and `pflow_is_charged`. The writer also
produces its existing object-class, mask, incidence, and event datasets and a
derived ROOT output. The deterministic baseline has no distribution parameters
in its prediction schema.

## Baseline boundary

The MDN work must preserve this deterministic task and its configured behavior.
A separate task and experiment config may reuse proxy/scaler behavior, but any
changes to preprocessing, proxy definitions, matching, prediction writing, or
the deterministic regression path require explicit regression evidence.
