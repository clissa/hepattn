# GLOW model data flow

GLOW operates on two interacting sets:

- **detector nodes**, representing tracks and topological calorimeter clusters;
- **particle queries**, representing learnable candidate-particle slots.

The architecture is therefore more than a linear encoder followed by four
independent prediction heads. In particular, intermediate constituent masks
control attention inside the decoder, while the final class and incidence
predictions are used to construct the input to the kinematic calibration.

```text
Tracks + topoclusters
          │
          ▼
  Transformer encoder
          │
          ├──────── encoded detector nodes ───────────────┐
          │                                               │
          ▼                                               │
600 learned particle queries                              │
          │                                               │
          ▼                                               │
┌─────────────────────────────────────────────────────────┤
│  Mask prediction → restricts cross-attention            │
│          │                                              │
│          ▼                                              │
│  Masked decoder layer                                   │
│          │                                              │
│          └── repeated for four layers ──────────────────┘
└─────────────────────────────────────────────────────────
          │
          ├──► PID and null classification
          │
          ├──► binary constituent mask
          │
          └──► fractional incidence matrix
                         │
                ┌────────┴────────┐
                │                 │
          charged proxy      neutral proxy
                └────────┬────────┘
                         │
       query embedding + proxy + weighted nodes
                         │
                         ▼
              kinematic calibration
                         │
                         ▼
                  PFlow particles
```

## 1. Detector inputs

For each event, tracks and topoclusters are combined into one sequence of
detector nodes. Tracks are stored first and topoclusters second, followed by
padding, but conceptually the Transformer processes the valid nodes as an
unordered set rather than as a time-ordered sequence.

Each node has 18 input features. Some, such as transverse momentum, eta, and
phi, are shared. Others are specific to the detector-object type:

- track features include impact parameters and coordinates extrapolated to the
  calorimeter;
- topocluster features include energy, number of cells, electromagnetic
  fraction, and geometric quantities;
- two explicit flags identify tracks and topoclusters.

An input network maps each node to a 256-dimensional embedding. A Fourier
positional encoding derived from the node eta and phi is added to this
representation. The relevant configuration is in
[`glow_baseline.yaml`](../../src/hepattn/experiments/atlas/configs/glow_baseline.yaml).

## 2. Transformer encoder

A six-layer Transformer encoder allows all detector nodes to exchange
information. Its output is a set of contextualized node embeddings: a track
embedding can contain information about compatible calorimeter activity, and a
topocluster embedding can contain information about nearby tracks and other
clusters.

These encoded node representations are retained for both the masked decoder
and the final constituent-association heads. The input embedding and encoder
flow are implemented in
[`maskformer.py`](../../src/hepattn/models/maskformer.py).

## 3. Learned particle queries

In parallel, the model defines a fixed set of 600 learned particle queries. A
query is initially only a trainable vector: it does not correspond to a
specific physical particle before processing an event.

During inference, the queries act as candidate-particle slots. The model learns
that:

- some queries should represent reconstructed particles;
- excess queries should be assigned to the null class;
- different active queries should represent different particles.

The number of queries is therefore an upper bound on the number of output
candidates, not the predicted particle multiplicity. The multiplicity is
determined by how many queries are ultimately classified as non-null.

## 4. Iterative masked decoder

The decoder contains four layers. Before each decoder layer, the model produces
two intermediate predictions for every query:

1. preliminary class logits;
2. a preliminary query-to-node constituent mask.

The mask has the conceptual shape

```text
[number of queries] × [number of detector nodes].
```

Each entry is obtained from learned projections of the query and node
embeddings followed by a dot product. It estimates whether a detector node is
associated with a particular query.

The predicted mask is thresholded and converted into an attention mask. In the
following decoder operation, each query concentrates its cross-attention on the
detector nodes currently associated with it. The reconstruction is therefore
iterative:

```text
query
  → preliminary constituent prediction
  → attention to selected detector nodes
  → updated query
  → refined constituent prediction
  → ...
```

The cross-attention is bidirectional in the current implementation. Queries are
updated using detector-node embeddings, and detector-node embeddings are also
updated using the queries. After each layer, both the candidate-particle and
detector-node representations are refined.

The mask calculation is implemented by `ObjectHitMaskTask` in
[`task.py`](../../src/hepattn/models/task.py), while the iterative attention
flow is implemented in
[`decoder.py`](../../src/hepattn/models/decoder.py).

## 5. Final prediction branches

After the four decoder layers, the final query and node embeddings feed three
main branches.

### 5.1 Particle classification and counting

Each query is classified into five physical classes plus a null class. This
head therefore performs three related functions:

- particle identification;
- selection of active query slots;
- implicit determination of the event's reconstructed-particle multiplicity.

A query is considered a valid reconstructed particle only when its most likely
class is one of the five physical classes. Queries assigned to the null class
are discarded.

### 5.2 Binary constituent mask

The final constituent mask represents a binary association between candidate
particles and detector nodes:

> Does this track or topocluster belong to this candidate particle?

It identifies the set of likely constituents for each candidate. It does not,
however, quantify how much of a detector node should be assigned to that
particle.

### 5.3 Fractional incidence matrix

The incidence matrix has the same query-by-node organization as the mask but a
different physical meaning:

> What fraction of this detector node is assigned to each candidate particle?

The model projects the query and detector-node embeddings, takes their dot
products, and applies a softmax over the query dimension. Consequently, the
predicted assignments for each valid detector node are normalized across the
candidate particles.

For example:

```text
                 particle 1   particle 2   particle 3
topocluster A        0.7          0.3          0.0
topocluster B        0.0          0.2          0.8
track C              1.0          0.0          0.0
```

The binary mask indicates which associations are active, whereas the incidence
matrix retains fractional information. The incidence calculation is
implemented by `IncidenceRegressionTask` in
[`task.py`](../../src/hepattn/models/task.py).

## 6. Charged and neutral proxy branches

Kinematic calibration is not an independent head running directly in parallel
with classification and incidence. It consumes their predictions to construct
a physics-motivated proxy particle for every query.

The predicted particle class first divides the queries into charged and neutral
branches.

### Charged proxy

For a query classified as charged:

- only detector nodes identified as tracks are considered;
- the incidence scores are used to select the most strongly associated track;
- the track transverse momentum, eta, and phi define the initial proxy
  kinematics;
- the proxy energy is constructed as `pT × cosh(eta)`.

### Neutral proxy

For a query classified as neutral:

- track nodes are excluded;
- incidence values are multiplied by the topocluster energies;
- the sum supplies the proxy energy;
- the direction is constructed from incidence- and energy-weighted
  topocluster quantities;
- the proxy transverse momentum is derived as `E / cosh(eta)`.

The two branches are combined into one five-component proxy representation:

```text
E, pT, eta, sin(phi), cos(phi).
```

This charged/neutral proxy construction is implemented by
`IncidenceBasedRegressionTask.get_proxy_feats()` in
[`task.py`](../../src/hepattn/models/task.py).

## 7. Final kinematic calibration

For every query, the regression network receives:

- the final query embedding;
- the charged or neutral physics proxy;
- a charged/neutral indicator;
- an incidence-weighted combination of the final detector-node embeddings.

In the deterministic baseline, the regression network predicts five offsets in
the scaled feature space. These offsets are added to the proxy:

```text
final kinematics = physics proxy + learned correction.
```

The calibrated output contains energy, transverse momentum, eta, sin(phi), and
cos(phi). Thus, the final regression combines a physics-motivated starting
point with a learned correction informed by the query and its associated
detector activity.

## 8. Training-time matching branch

The model predicts an unordered set of queries, while the target particles are
also unordered. During training, an optimal bipartite matcher assigns predicted
queries to truth particles before the losses are evaluated.

For the final predictions, the matching cost combines enabled contributions
from classification, constituent mask, incidence, and kinematic regression.
Classification and mask predictions are also supervised at intermediate
decoder layers. Intermediate incidence and kinematic regression are disabled in
the baseline configuration.

This matching procedure is part of training only. At inference time there is no
truth matching: non-null queries directly form the reconstructed PFlow particle
set.

## Implication for architecture diagrams

Showing `Mask`, `PID`, `Incidence`, and `Kinematics` as four equivalent and
independent output heads is convenient but incomplete. A more faithful diagram
should show:

- a feedback arrow from the mask prediction to the masked decoder;
- PID and incidence as final prediction branches;
- PID, incidence, and detector-node information converging into a physics
  proxy;
- the proxy and query embedding entering the kinematic calibration;
- the null PID class removing unused query slots from the final output.

It is also more accurate to describe the core idea as:

> **The input event is represented as an unordered set of tracks and
> topoclusters, and the reconstructed particles form an unordered set of
> learned queries.**
