# ATLAS MDN promising experiment queue

## Purpose and status

This note records the small-scale ATLAS MDN training observations discussed in
August 2026, the relevant implementation details, and the consolidated priority
queue for subsequent experiments. It is intentionally a planning document: no
new configuration below should combine unrelated changes, and each result should
be compared with a frozen reference configuration.

The two AdamW follow-up runs were incomplete when this note was written. Their
partial curves must not be used to choose the final reference model.

## Experiment map

| Comet run | Configuration/change | Optimizer | Status |
|---|---|---|---|
| `loud_cuisine_2246` | one-component MDN, deterministic weight 6 | Lion | complete comparison run |
| `increased_dish_15` | one-component MDN, deterministic weight 6 | AdamW | complete comparison run |
| `fiscal_mole_2180` | one-component MDN, deterministic weight 60 | Lion | complete comparison run |
| `modern_guava_9840` | three-component MDN, deterministic weight 6 | Lion | complete comparison run |
| additional AdamW runs | high deterministic weight or three components | AdamW | ongoing/incomplete |

Current local configuration files include:

- `src/hepattn/experiments/atlas/configs/mdn_lion_base.yaml`
- `src/hepattn/experiments/atlas/configs/mdn_adamw_base.yaml`
- `src/hepattn/experiments/atlas/configs/mdn_lion_k3.yaml`
- `src/hepattn/experiments/atlas/configs/mdn_lion_detw60.yaml`
- `src/hepattn/experiments/atlas/configs/mdn_adamw_k3.yaml`
- `src/hepattn/experiments/atlas/configs/mdn_adamw_detw60.yaml`

The filename stem, run name, and output-directory basename are normalized for
these historical configurations. Their scientific settings are unchanged.

## Plot-derived observations

The screenshots themselves are not stored in this repository. The values and
trends visible in the supplied Comet panels are summarized below so that the
experimental conclusions remain reviewable.

### Plot: aggregate train and validation loss

At approximately step 1,619:

| Run | Validation loss |
|---|---:|
| `modern_guava_9840`, three components | **3.595** |
| `increased_dish_15`, AdamW | 3.905 |
| `loud_cuisine_2246`, Lion | 3.923 |
| `fiscal_mole_2180`, deterministic weight 60 | 8.151 |

The ordinary AdamW and Lion runs are effectively tied at this resolution. The
high-weight loss is not numerically comparable because one contribution was
multiplied by ten. The three-component total is lower, but the component plots
show that this is principally an MDN likelihood gain rather than a universal
reconstruction gain.

Several runs have large early transients. With about 1,600 total steps and
`pct_start: 0.05`, OneCycleLR reaches its peak after only about 80 steps. This
supports testing a slower warmup and a lower peak learning rate independently.

### Plot: MDN NLL and deterministic L1

At approximately step 1,619:

| Run | Validation MDN NLL | Logged deterministic L1 | Loss weight | Approx. raw deterministic L1 |
|---|---:|---:|---:|---:|
| three-component Lion | **-3.568** | 0.6952 | 6 | 0.1159 |
| high-weight Lion | -3.357 | 5.296 | 60 | **0.0883** |
| AdamW | -3.202 | 0.5769 | 6 | 0.0962 |
| Lion | -3.131 | 0.6146 | 6 | 0.1024 |

The logged deterministic loss is already weighted in
`IncidenceBasedMixtureRegressionTask.loss`. Dividing by its configured weight is
therefore necessary for a cross-run comparison.

Conclusions:

- Weight 60 reduces raw deterministic error by about 14% relative to Lion at
  weight 6, about 8% relative to AdamW, and about 24% relative to the
  three-component Lion run.
- Weight 60 also improves MDN NLL relative to both ordinary one-component runs.
  It is not merely exchanging energy/`pT` likelihood for angular accuracy.
- Three components produce the best density likelihood, but deterministic
  point regression becomes worse.
- A negative continuous-density NLL is valid; more negative is better within
  the same target scaling and likelihood definition.

### Plot: validation mask BCE and Dice loss

At approximately step 1,619:

| Run | Mask BCE | Mask Dice loss |
|---|---:|---:|
| high-weight Lion | **0.1982** | **0.6337** |
| AdamW | 0.2120 | 0.6526 |
| Lion | 0.2113 | 0.6621 |
| three-component Lion | 0.2173 | 0.6728 |

High deterministic weighting improves both mask objectives despite not changing
their explicit weights. Three components slightly degrade both.

### Plot: exact match, recall, and purity

At approximately step 1,619:

| Run | Exact match | Recall | Purity |
|---|---:|---:|---:|
| high-weight Lion | **0.01054** | **0.3378** | **0.2818** |
| AdamW | 0.00877 | 0.3039 | 0.2624 |
| Lion | 0.00653 | 0.2791 | 0.2592 |
| three-component Lion | 0.00553 | 0.2661 | 0.2443 |

Relative to ordinary Lion, high deterministic weighting improves recall by
about 21%, purity by about 9%, and exact match by about 61%. Exact match remains
very low in absolute terms, so mask association still has substantial headroom.

### Plot: incidence KL divergence

At approximately step 1,619:

| Run | Validation incidence KL |
|---|---:|
| high-weight Lion | **0.01690** |
| AdamW | 0.01768 |
| three-component Lion | 0.01805 |
| Lion | 0.01809 |

Three components leave incidence essentially unchanged. High deterministic
weighting improves it by roughly 7% relative to ordinary Lion.

## Joint interpretation

### Strongest completed direction: deterministic weight 60

The high-weight Lion run is the strongest holistic result so far. Its aggregate
loss appears worse only because the deterministic term is rescaled. After
accounting for that scale, it improves every supplied validation category:

- deterministic regression;
- one-component MDN NLL;
- mask BCE and Dice;
- mask recall, purity, and exact match;
- incidence KL.

The likely mechanism is improved shared representation learning. Regression
constructs proxy inputs from detached classification and incidence predictions,
so its loss does not directly train those predictions through the proxy path.
It does, however, train the shared query features, shared node features, encoder,
decoder, and regression head. Better shared representations can therefore help
all tasks through their own losses.

The original deterministic weight of 6 was probably too small relative to the
classification and mask losses, which are repeated at every decoder stage.

### Three components: likelihood specialist, not current overall winner

Three components lower MDN NLL substantially but worsen deterministic L1 and
most mask metrics. The model trains mixture likelihood, while its point estimate
and regression matching use the probability-weighted mixture mean. A better
multimodal density does not guarantee a better mean prediction.

Before increasing mixture capacity, inspect:

- `val/final_regression_mdn_point_l1`;
- `val/final_regression_point_l1`;
- energy and `pT` response and resolution;
- component occupancy or collapse;
- predicted scale calibration and interval coverage;
- tail performance on a fixed evaluation sample.

Five components and a Student-t mixture are deferred until the three-component
model demonstrates a benefit beyond NLL.

### Optimizer conclusion remains provisional

AdamW and Lion are nearly tied at the shared learning-rate schedule. This does
not prove that their individually tuned optima are equal. Finish the ongoing
AdamW high-weight and three-component runs before selecting the reference
optimizer. Do not interpret their partial trajectories as final results.

## Preliminary model and training notes

### Data and batching

- Training uses 12,000 sampled events per epoch and validation uses 6,000.
- The per-device batch size is 112 on three GPUs, giving a nominal global batch
  of 336.
- `use_distributed_sampler: false` is configured.
- Each rank constructs a `RandomSampler` when `num_*_per_epoch` is set. The
  validation subset can therefore change between epochs, adding noise to curves
  and checkpoint selection.
- The data limits are 800 input nodes and 600 target/query slots. Events beyond
  these configured limits are filtered.
- `e`, `pT`, and `eta` are scaled; `sinphi` and `cosphi` are not scaled.

### Architecture

- Node inputs are projected to dimension 256 and receive Fourier positional
  encoding in `eta` and `phi`.
- The encoder has six full-attention layers, HybridNorm, value residuals,
  FlashAttention varlen, 16 heads, and eight register tokens.
- The decoder has four layers, 600 learned queries, 16 heads, mask-guided
  attention, and bidirectional query/node cross-attention.
- The four tasks are object classification, object-hit mask prediction,
  incidence regression, and incidence-based MDN regression.

### Supervision and matching

- Classification and object-hit masks are produced and supervised at all four
  decoder layers and again at the final output.
- Incidence and MDN regression are final-only.
- The total optimized loss is a plain sum over every task loss at every
  supervised layer; there is no automatic normalization across tasks or depth.
- Hungarian matching is performed independently for every supervised output
  layer.
- Final matching combines classification CE cost 2, mask Dice cost 1,
  incidence KL cost 1, and regression point-L1 cost 10.
- Training combines classification CE 2, mask BCE 5, mask Dice 1, incidence KL
  1, MDN NLL 1, and deterministic L1 6 or 60. Classification and mask terms are
  repeated across decoder stages.
- Matching weights and training-loss weights are distinct experimental knobs.

### MDN behavior

- The MDN applies a diagonal Gaussian mixture to scaled `e` and `pT`.
- `eta`, `sinphi`, and `cosphi` use deterministic mean L1.
- `sinphi` and `cosphi` are optimized independently; there is no explicit unit
  circle or angular-distance penalty.
- MDN means are offsets from incidence-based proxy features.
- Scales are positive through softplus with a floor of `1e-3` and initial scale
  `1e-1`.
- For `K` mixture components, the regression head width is
  `K * (1 + 2 * 2) + 3 = 5K + 3`; changing `num_components` requires the
  corresponding output width.
- The MDN calculation and loss are cast to float32 even though training uses
  mixed bfloat16.

### Optimizer and schedule

- Only AdamW and Lion are currently supported by `ModelWrapper`.
- Both current baselines use initial LR `2e-6`, peak LR `2e-4`, final LR
  `2e-6`, weight decay `1e-2`, and OneCycleLR.
- `pct_start: 0.05` gives a very short warmup and is consistent with the early
  loss spikes.
- Gradient clipping is configured at 0.1. Gradient norms are not included in
  the supplied panels, so it is unknown how frequently clipping is active.
- The commented multi-task-learning path is not currently runnable as a config
  option: `mtl: true` would call a commented-out optimization method.

### Implementation cautions relevant to experiments

- Aggregate `val/loss` is comparable only when all loss definitions and weights
  are the same.
- Checkpoints monitor aggregate `val/loss`; this is unsuitable for selecting
  between weight-6 and weight-60 objectives.
- The object-hit task currently assigns `has_intermediate_loss` from
  `mask_attn`, overriding the constructor argument. Turning off only the YAML
  `has_intermediate_loss` flag would not isolate intermediate mask supervision.
- Disabling decoder `mask_attention` is a valid clean ablation while retaining
  mask prediction and its losses.

## Comparison protocol before expanding the queue

1. Finish the ongoing AdamW runs at the same step/epoch budget.
2. Select the reference using raw/task metrics and downstream physics metrics,
   not aggregate loss across differently weighted objectives.
3. Use high-weight Lion provisionally. Switch to high-weight AdamW only if the
   completed run wins consistently on deterministic error, MDN NLL, mask
   metrics, incidence, and downstream reconstruction.
4. Use a fixed validation sample. The existing config-only option is
   `num_val_per_epoch: null`, which evaluates the full validation set. A fixed
   6,000-event subset would require a small loader change.
5. Freeze the reference config, seed, data, sample budget, and evaluation cuts
   for the first screening pass.
6. Compare final-layer task metrics and physics plots. Save aggregate loss only
   as an optimization diagnostic.
7. Repeat finalists with at least two additional seeds before drawing a strong
   conclusion.

Useful selection metrics include:

- raw deterministic L1 (logged weighted value divided by its weight);
- `final_regression_point_l1` and `final_regression_mdn_point_l1`;
- energy and `pT` response, core resolution, and tails;
- angular and `eta` resolution;
- class-wise efficiency, fake rate, and duplicate rate;
- mask recall, purity, Dice, and exact match;
- incidence KL;
- MDN calibration, component usage, and coverage.

## Consolidated next-config priority queue

Every entry below is one targeted experimental feature. All unspecified fields
must remain identical to the selected high-weight reference.

The two-GPU operational reference is `qs00_ref_detw60`. Every queue config uses
Lion, two devices, a per-device batch size of 80, gradient accumulation of two,
and 22 epochs. The twelve alternatives below are derived independently from
that reference.

### 1. Deterministic weight 120 (`qs01_detw120`)

Change only:

```yaml
deterministic_loss_weight: 120
```

Rationale: increasing 6 to 60 improved every supplied metric, so a higher
bracket tests whether deterministic supervision remains limiting. If 120
regresses, test 30 as the lower bracket around the successful value 60.

Primary metrics: raw deterministic L1, MDN NLL, point L1, masks, incidence, and
gradient stability.

### 2. Slower OneCycle warmup (`qs02_warm20`)

Change only:

```yaml
pct_start: 0.20
```

Rationale: the current schedule reaches peak LR in roughly 80 steps and several
runs show large early transients. This tests warmup duration without changing
the peak or final LR.

Primary metrics: early loss stability, final task metrics, and convergence by
the fixed epoch budget.

### 3. Lower peak learning rate (`qs03_lrmax1e4`)

Change only:

```yaml
max: 1.0e-4
```

Keep `pct_start: 0.05` in this experiment so that it remains distinct from the
warmup test.

Rationale: the present `2e-4` peak may be too aggressive, particularly for
Lion. AdamW and Lion should eventually receive separate one-factor LR tuning.

### 4. Geometry-aware deterministic loss (`qs04_geomphi`)

Implement one coherent replacement for the current three-field mean L1:

- scaled-`eta` L1 with internal weight `1/3`;
- `1 - cos(delta_phi)` with internal weight `2/3`;
- `(sinphi^2 + cosphi^2 - 1)^2` with internal weight `0.05`.

The task exposes these as separate loss components. The fixed weights are part
of the geometry feature; the config selects only
`deterministic_loss_mode: geometry`.

Keep the MDN likelihood, optimizer, matching, architecture, and data unchanged.
Calibrate the total deterministic contribution against the successful weight-60
reference.

Rationale: stronger deterministic supervision helps, while independent
`sinphi`/`cosphi` L1 does not directly optimize angular distance or valid unit
vectors. This requires a focused code change and unit tests.

### 5. Focal object-hit mask loss (`qs05_maskfocal`)

Replace mask BCE only:

```yaml
focal_gamma: 2.0
losses:
  mask_focal: 5.0
  mask_dice: 1.0
```

Leave matching costs unchanged.

Rationale: object-node assignment is highly imbalanced and absolute exact-match,
recall, and purity remain low. Focal loss is already supported by the code.

### 6. Intermediate incidence supervision (`qs06_incaux`)

Change the incidence task only:

```yaml
has_intermediate_loss: true
```

Keep incidence loss and matching weights at one.

Rationale: incidence is central to the regression proxy but currently receives
only final-layer supervision. This coherent feature inherently adds incidence
predictions, matching contributions, and losses at decoder stages.

### 7. Lower classification null weight (`qs07_null02`)

Change only the classification task:

```yaml
null_weight: 0.2
```

Rationale: 600 queries create many null slots, and classification is supervised
at five stages. Lower null pressure may improve recall, but fake and duplicate
rates are mandatory counter-metrics.

### 8. Reduce regression cost in Hungarian matching (`qs08_matchreg3`)

Change only the MDN regression matching cost:

```yaml
cost_weight: 3.0
```

Do not change MDN or deterministic training-loss weights.

Rationale: early point predictions may dominate final matching at the current
weight 10. A lower nonzero weight lets class, mask, and incidence structure have
more influence without removing useful kinematics entirely.

### 9. Disable hard mask-guided decoder attention (`qs09_nomaskattn`)

Change only:

```yaml
decoder:
  mask_attention: false
```

Retain the mask head and mask losses.

Rationale: low early mask quality may hide relevant nodes from queries. This
tests routing without conflating it with mask supervision.

### 10. Six decoder layers (`qs10_dec6`)

Change only:

```yaml
num_decoder_layers: 6
```

Rationale: two extra refinement stages may improve association and object
formation. Additional intermediate classification and mask terms are an
inherent consequence of this architectural feature. Record memory, step time,
and final-layer metrics, because aggregate loss will contain more terms.

### 11. Deterministic weight 30 (`qs11_detw30`)

Change only:

```yaml
deterministic_loss_weight: 30
```

Rationale: this lower bracket tests whether the gains at weight 60 require the
full increase, and complements the weight-120 upper bracket.

### 12. Initial MDN scale 0.3 (`qs12_scaleinit03`)

Change only:

```yaml
initial_scale: 3.0e-1
```

Rationale: a broader initial one-component Gaussian probes scale calibration
without increasing mixture capacity. It remains lowest priority because the
available plots do not directly motivate the direction.

## Deferred or conditional experiments

These remain scientifically interesting but are lower priority after reviewing
the component plots:

- **Five MDN components (`output_size: 28`):** defer until three components
  improve point estimates, calibration, or physics tails rather than NLL alone.
- **Student-t mixture likelihood:** defer for the same reason; it targets heavy
  tails but may further optimize density without improving reconstruction.
- **Cosine decay with warmup:** requires scheduler implementation and should
  follow the two supported OneCycle tests.
- **Encoder depth or width:** decoder/refinement and loss-balance experiments
  are better motivated by the current plots.
- **Dropout, stochastic depth, or LayerScale:** consider if longer runs show
  overfitting or instability; the current curves do not establish that yet.
- **Additional optimizers:** the wrapper currently supports only AdamW and Lion.
  Tune their learning rates before adding another optimizer implementation.
- **Three components plus weight 60:** this is a factorial interaction test,
  not a clean first-pass main-effect experiment. Consider it only after the
  completed AdamW comparisons and three-component point-metric review.

## Execution order in three-job waves

Run the two-GPU reference and all alternatives in fixed priority order:

1. `qs00_ref_detw60`, `qs01_detw120`, `qs02_warm20`;
2. `qs03_lrmax1e4`, `qs04_geomphi`, `qs05_maskfocal`;
3. `qs06_incaux`, `qs07_null02`, `qs08_matchreg3`;
4. `qs09_nomaskattn`, `qs10_dec6`, `qs11_detw30`;
5. `qs12_scaleinit03`.

Do not fold a winner into the next pair during this initial screening pass.
Every queued run should differ from the same frozen reference by only its named
feature. After the main-effect screen, confirm the strongest changes with extra
seeds and only then test selected interactions.

## Source locations

- Training wrapper and total loss aggregation:
  `src/hepattn/models/wrapper.py`
- MaskFormer forward path, per-layer matching, and loss construction:
  `src/hepattn/models/maskformer.py`
- Decoder and mask-guided bidirectional attention:
  `src/hepattn/models/decoder.py`
- Task definitions, losses, matching costs, MDN, and metrics:
  `src/hepattn/models/task.py`
- Loss and cost functions:
  `src/hepattn/models/loss.py`
- Encoder, register tokens, HybridNorm, value residual, and LayerScale support:
  `src/hepattn/models/transformer.py`
- Data sampling and loaders:
  `src/hepattn/experiments/atlas/pflow_data.py`
- Baseline and experiment configurations:
  `src/hepattn/experiments/atlas/configs/`
