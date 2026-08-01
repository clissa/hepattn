# Quick-search execution plan

## Goal

Create a clean `quick-search` branch that preserves the current MDN experiment
work in modular commits, normalizes historical config/run naming, defines a
ranked twelve-feature search queue based on the `fiscal_mole` Lion +
deterministic-weight-60 result, and provides a fail-loud launcher that runs up to
three two-GPU trainings concurrently until the queue is exhausted.

This file is an execution plan only. The planning session that created it must
not create the branch, edit experiment configs or training code, commit, push,
or launch GPU jobs.

## Current authoritative state

- Current branch: `mdn_dev`.
- The worktree is dirty and all existing changes belong to the user.
- Pending files currently include:
  - modified `base_MDN-AdamW_jz1234.yaml`;
  - untracked `base_MDN-Lion_jz1234.yaml`;
  - untracked Lion/AdamW three-component configs;
  - untracked Lion/AdamW deterministic-weight-60 configs;
  - untracked `configs/configs_queue/promising.md`.
- The completed reference result is `fiscal_mole`: Lion, one MDN component,
  deterministic loss weight 60.
- The present high-weight Lion file has a `copy` suffix.
- The present AdamW high-weight file selects AdamW but still has a Lion name.
- `promising.md` currently contains ten ranked experiments. Its current path is
  `src/hepattn/experiments/atlas/configs/configs_queue/promising.md`.
- The intended runtime is the repository venv at `.hepattn`, not Pixi.
- Both `todo/` and `scratch/` are ignored by `.gitignore`. The two explicitly
  requested artifacts in those directories must be staged with exact-path
  force-adds; do not change the ignore policy for every todo or scratch file.

Before acting, re-run `git status --short`, inspect the actual diffs, and treat
the then-current worktree as authoritative. Do not discard or rewrite changes
that appeared after this plan was written.

## Non-negotiable experimental rules

1. Each alternative differs from one frozen search reference by exactly one
   named scientific feature. Mechanically required companion edits, such as an
   MDN output width, are part of that single feature.
2. Do not combine more feature changes in one alternative, e.g. a scheduler change with an optimizer, loss, or architecture change.
3. All queue runs use Lion and deterministic weight 60 unless that field is the
   targeted feature.
4. All queue runs use two GPUs. Because this changes distributed training from
   the original three-GPU `fiscal_mole` run, rerun the unmodified scientific
   reference with two GPUs as `qs00_ref_detw60`.
5. Change per-device batch size to 80 with `accumulate_grad_batches: 2` and `max_epochs: 22` for the first search. This gives a two-GPU
   nominal global batch of 160 and an effective batch of 320, for a total of ~1650 optimizer steps and ~528k processed events; the new `qs00` reference controls for that
   operational change. Do not silently scale LR or batch size.
6. Keep data paths, seed, sample counts, precision, matching, and all unspecified
   fields identical to the reference.
7. Config filename stem, top-level `name`, Comet run name, output-directory
   basename, and log basename must use the same short experiment ID.
8. Do not launch training while implementing this plan. Finish and verify the
   branch, configs, and launcher, then hand the launch command to the user.

## Naming convention

Use lowercase snake-case IDs that are short but interpretable:

```text
filename:          <experiment_id>.yaml
top-level name:    <experiment_id>
Comet display name:<experiment_id>  # linked by the existing CLI/logger
default_root_dir:  /home/lclissa/projects/hepattn/results/quick-search/<experiment_id>
stdout/stderr log: /home/lclissa/projects/hepattn/logs/<experiment_id>.log
```

Queue IDs use a zero-padded priority prefix:

```text
qs00_ref_detw60
qs01_detw120
...
qs12_scaleinit03
```

The prefix makes filesystem, Comet, launcher, and result ordering agree.

## Phase 1: create the branch

1. Confirm the current branch and dirty state.
2. Check whether `quick-search` already exists locally.
3. Run `git switch -c quick-search` while preserving all
   pending changes.
4. Verify that all pending files are still present after switching.

Success check:

```text
git branch --show-current  -> quick-search
git status --short         -> same pending user work, nothing lost
```

## Phase 2: modularly commit the current pending work

Use the repository `commit` skill for every commit. Inspect each staged diff
before committing and never stage unrelated files. The exact split may adapt if
the worktree changed, but the current logical split is:

### Commit 1: record the short-run AdamW and Lion baselines

Stage only:

- `base_MDN-AdamW_jz1234.yaml`
- `base_MDN-Lion_jz1234.yaml`

Suggested subject:

```text
Add short-run AdamW and Lion MDN configs
```

### Commit 2: record completed optimizer and MDN variants

Stage only the four current high-weight/three-component files, retaining their
current names in this historical commit:

- Lion, three components;
- Lion, deterministic weight 60;
- AdamW, three components;
- AdamW, deterministic weight 60.

Suggested subject:

```text
Add MDN optimizer follow-up configs
```

### Commit 3: record the experiment analysis and queue notes

Stage only:

- `src/hepattn/experiments/atlas/configs/configs_queue/promising.md`
- `todo/quick-search.md`

Because `todo/` is ignored, use `git add -f -- todo/quick-search.md` for that
exact file. Confirm it appears in `git diff --cached`; do not force-add the
directory.

Suggested subject:

```text
Document the ATLAS MDN quick search
```

After each commit, inspect `git status --short` and `git show --stat --oneline
HEAD`. Do not claim the pending work is committed until every original pending
file is either committed or explicitly identified as unrelated user work.

## Phase 3: normalize existing config and Comet names

Rename the six historical short-run configs to stable names and make the
filename stem, top-level `name`, and output basename agree:

| Scientific config | New ID |
|---|---|
| Lion baseline | `mdn_lion_base` |
| AdamW baseline | `mdn_adamw_base` |
| Lion, deterministic weight 60 | `mdn_lion_detw60` |
| AdamW, deterministic weight 60 | `mdn_adamw_detw60` |
| Lion, three components | `mdn_lion_k3` |
| AdamW, three components | `mdn_adamw_k3` |

For each config:

```text
filename            = <ID>.yaml
name                = <ID>
default_root_dir    = /home/lclissa/projects/hepattn/results/quick-search/<ID>
```

Search the repository for references to the old filenames and update obvious
callers or comments in the same naming commit. Importantly, rename all `results/quick-search/` subfolders corersponding to past experiments accordingly.
Backward-compatible aliases are
not required. Do not alter optimizer, dataset, loss, architecture, or schedule
while normalizing names.

Parse every renamed YAML and programmatically assert:

- filename stem equals `name`;
- output-directory basename equals `name`;
- expected optimizer and targeted feature remain unchanged.

Commit this phase separately with the `commit` skill.

Suggested subject:

```text
Normalize ATLAS MDN experiment names
```

## Phase 4: expand and implement the ranked search queue

### Search reference

Create:

```text
src/hepattn/experiments/atlas/configs/configs_queue/qs00_ref_detw60.yaml
```

Copy the normalized `mdn_lion_detw60` scientific configuration, then make only
the common search-operational edits:

- `trainer.devices: 2`;
- `name: qs00_ref_detw60`;
- queue-specific `default_root_dir`.

All alternatives must be derived from this file, not from one another.

### Final top-twelve priority

Extend `promising.md` from ten to twelve priorities and ensure the prose,
deferred list, execution order, and filenames agree with this final queue:

| Priority | ID | Single targeted feature | Exact intended change |
|---:|---|---|---|
| 1 | `qs01_detw120` | stronger deterministic weight | `deterministic_loss_weight: 120` |
| 2 | `qs02_warm20` | slower OneCycle warmup | `pct_start: 0.20` |
| 3 | `qs03_lrmax1e4` | lower peak LR | `max: 1.0e-4` |
| 4 | `qs04_geomphi` | geometry-aware deterministic loss | select the new geometry loss mode |
| 5 | `qs05_maskfocal` | focal mask training loss | BCE -> focal, gamma 2; Dice unchanged |
| 6 | `qs06_incaux` | intermediate incidence supervision | incidence `has_intermediate_loss: true` |
| 7 | `qs07_null02` | lower classification null pressure | classification `null_weight: 0.2` |
| 8 | `qs08_matchreg3` | lower regression matching cost | regression `cost_weight: 3.0` |
| 9 | `qs09_nomaskattn` | disable hard mask-guided decoding | decoder `mask_attention: false` |
| 10 | `qs10_dec6` | deeper decoder | `num_decoder_layers: 6` |
| 11 | `qs11_detw30` | lower bracket around weight 60 | `deterministic_loss_weight: 30` |
| 12 | `qs12_scaleinit03` | broader initial MDN scale | `initial_scale: 3.0e-1` |

The first ten preserve the evidence-updated order in `promising.md`. Priority 11
turns the documented conditional weight-30 candidate into a runnable bracket.
Priority 12 probes MDN scale initialization without adding mixture capacity; it
is lower priority because no supplied calibration plot directly motivates its
direction.

### Geometry-aware loss support

`qs04_geomphi` is not currently expressible by YAML alone. Implement the
smallest backward-compatible task option needed to make it runnable:

- preserve the current L1 behavior as the default;
- add an explicit deterministic loss mode for geometry-aware supervision;
- apply L1 or SmoothL1 to scaled `eta`;
- apply a wrapped/angular loss such as `1 - cos(delta_phi)` using predicted and
  target `atan2(sinphi, cosphi)`;
- add a small, explicit unit-circle penalty for predicted sine/cosine norm;
- expose each component clearly enough to test and interpret;
- do not change MDN fields, matching cost, proxy construction, or other tasks;
- add focused deterministic CPU tests covering angular wrapping, unit-circle
  behavior, masking of invalid targets, default backward compatibility, and
  finite gradients.

Choose and document the internal component weights before generating the config.
They are a coherent part of the geometry-loss feature, but must not silently
depend on unrelated queue settings.

### Generate and audit the configs

Create exactly twelve alternative YAML files plus the `qs00` reference under
the existing `configs/configs_queue/` directory. For every file:

- use `trainer.devices: 2`;
- use Lion;
- use deterministic weight 60 except priorities 1 and 11;
- preserve following specs about batch size and all data/sample settings: `batch_size: 80`, `accumulate_grad_batches: 2`, `max_epochs: 22`;
- use the naming convention above;
- ensure the file differs from `qs00` only in name, output directory, and its
  one targeted feature;
- include mechanically required companion edits only when necessary;
- do not use YAML inheritance unless the existing Lightning/jsonargparse path
  is first proven to support it reliably.

Verification:

1. Parse all thirteen YAML files with the repository environment.
2. Instantiate or CLI-validate each config without starting training, where
   practical.
3. Assert unique names and output directories.
4. Assert `devices == 2` for all thirteen.
5. Produce a machine-readable diff summary against `qs00` and manually confirm
   each scientific delta.
6. Run the focused geometry-loss tests.
7. Run the existing MDN regression test file.
8. Run Ruff only on changed Python files; do not format unrelated files.

Commit the geometry support, tests, expanded notes, reference, and twelve queue
configs after they pass. If the implementation is large enough to obscure
review, use two commits: geometry-loss support/tests first, then queue docs and
configs. Use the `commit` skill and report the actual split.

Suggested subjects:

```text
Add geometry-aware MDN regression loss
Add prioritized ATLAS quick-search configs
```

## Phase 5: create the three-at-a-time launcher

Prefer a Bash launcher because the required operation is a small number of
shell pipelines with `tee`, environment variables, background jobs, and wave
barriers. Create:

```text
scratch/quick-search.sh
```

### Required queue order

Hard-code or declaratively list the thirteen configs in this order:

```text
qs00_ref_detw60
qs01_detw120
qs02_warm20
qs03_lrmax1e4
qs04_geomphi
qs05_maskfocal
qs06_incaux
qs07_null02
qs08_matchreg3
qs09_nomaskattn
qs10_dec6
qs11_detw30
qs12_scaleinit03
```

Process the list in waves of at most three. The first four waves have three
jobs and the final wave has one job. Never launch the next wave until every job
in the current wave exits.

### Required resource mapping

Within every wave:

| Slot | GPUs | Master port |
|---:|---|---:|
| 1 | `0,1` | `29500` |
| 2 | `2,3` | `29501` |
| 3 | `4,5` | `29502` |

Ports may be reused only after the wave barrier confirms the prior processes
have exited. Allow the base port to be overridden by an environment variable,
while preserving distinct slot offsets.

### Command and logging behavior

Run from:

```text
/home/lclissa/projects/hepattn/src/hepattn/experiments/atlas
```

Each job must be equivalent to:

```bash
MASTER_ADDR=127.0.0.1 \
MASTER_PORT=<slot_port> \
CUDA_VISIBLE_DEVICES=<slot_gpu_pair> \
python -u main.py fit -c configs/configs_queue/<experiment_id>.yaml \
2>&1 | tee /home/lclissa/projects/hepattn/logs/<experiment_id>.log
```

Use the `.hepattn` Python interpreter explicitly or activate that environment
in a controlled way. Create the log directory if absent. Use `set -euo pipefail`
and ensure a training failure is not hidden by `tee`.

The launcher must:

- print each start datetime, experiment ID, config, GPU pair, port, and log before launch;
- verify all configs exist before starting the first job;
- fail if duplicate names, ports, or output directories are detected;
- record and check every background PID;
- wait for all jobs in a wave even if one fails, then stop before the next wave
  with a nonzero exit and a clear failure summary;
- handle `SIGINT`/`SIGTERM` by terminating its child process groups and waiting,
  so distributed workers are not orphaned;
- include a `--dry-run` mode that prints all five waves without launching them;
- avoid deleting or overwriting checkpoints;
- make log overwrite behavior explicit. Prefer failing when a target log already
  exists unless the user supplies an explicit overwrite/resume choice.

Do not add automatic checkpoint resume or result-based pruning in this first
launcher; those are separate workflow features.

Verification:

```text
bash -n scratch/quick-search.sh
bash scratch/quick-search.sh --dry-run
```

Review the dry-run output to prove:

- slot 1 always receives GPUs 0,1;
- slot 2 always receives GPUs 2,3;
- slot 3 always receives GPUs 4,5;
- simultaneous jobs have different ports;
- all thirteen configs appear once in priority order;
- every log basename matches the config `name`.

Commit the launcher separately with the `commit` skill.

Because `scratch/` is ignored, use
`git add -f -- scratch/quick-search.sh` for that exact launcher only. Confirm it
appears in the staged diff; do not force-add other scratch files.

Suggested subject:

```text
Add batched ATLAS quick-search launcher
```

## Final completion audit

Before declaring the plan complete, verify all of the following from the current
repository state:

- branch is `quick-search`;
- original pending work is preserved in modular commits;
- the naming-only commit contains no scientific changes;
- historical config filename, name, and output basename are consistent;
- `promising.md` contains exactly twelve numbered priorities in the final order;
- `qs00` plus exactly twelve alternatives exist;
- all thirteen queue YAMLs parse and request two devices;
- alternatives have only their documented scientific delta from `qs00`;
- geometry mode has focused passing tests and legacy L1 remains the default;
- launcher passes syntax and dry-run checks;
- launcher assigns GPU pairs and ports exactly as required;
- launcher is not actually run during implementation;
- `git ls-files` proves both ignored-path deliverables (`todo/quick-search.md`
  and `scratch/quick-search.sh`) are tracked;
- git status contains no unexplained changes;
- commit history shows the intended modular sequence;
- nothing is pushed unless the user separately asks.

Summarize files, commits, verification commands, skipped GPU execution, and the
single command the user should run when ready.
