# ATLAS MDN quick-search round 2

This round repeats the strongest round-1 directions with the complete JZ1--4
training file, a memory-margined per-device batch, and a longer epoch budget.

| Config | Round-1 parent | Additional feature |
| --- | --- | --- |
| `qs13_ref_detw60` | `qs00_ref_detw60` | reference |
| `qs14_null02` | `qs07_null02` | classification null weight 0.2 |
| `qs15_incaux` | `qs06_incaux` | intermediate incidence supervision |
| `qs16_detw30` | `qs11_detw30` | deterministic loss weight 30 |
| `qs17_matchreg3` | `qs08_matchreg3` | regression matching cost 3 |
| `qs18_null02_incaux` | `qs07_null02` | null weight 0.2 plus intermediate incidence supervision |

Every config uses `JZ1234_train.root`, `batch_size: 96`, two devices,
`accumulate_grad_batches: 2`, `num_train_per_epoch: 15360`, and 42 epochs.
The batch is deliberately uniform across configurations: the observed 46.6 GB
usage at batch 112 left too little margin for the two configurations with
intermediate incidence supervision. All other settings are inherited from the
named round-1 parent.

## Training-budget comparison

`use_distributed_sampler: false` means each of the two ranks independently
draws the configured training events per epoch. The total processed examples below are
nominal: the two rank-local samples can overlap.

| Round | Batch/rank | Batches/rank/epoch | Optimizer steps/epoch | Epochs | Optimizer steps | Processed examples/rank | Nominal total processed examples |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 80 | 150 | 75 | 22 | 1,650 | 264,000 | 528,000 |
| 2 | 96 | 160 | 80 | 42 | 3,360 | 645,120 | 1,290,240 |

Round 2 therefore processes 2.44x more nominal examples and has 2.04x more
optimizer steps per configuration. Its effective global batch per optimizer
step rises from 320 to 384 examples.

The campaign has six configurations in two three-job waves, versus 13
configurations in five waves for round 1. Its 6,720 wave-adjusted optimizer
steps target the round-1 8,250-step wall-time budget if a batch-96 optimizer
step costs about 1.23x a batch-80 step. This is a planning estimate rather
than a measured runtime guarantee; it trades the unused final round-1 wave
capacity for substantially greater data exposure and convergence information
per configuration.

## Launch

From the repository root, inspect the planned two-wave launch with:

```shell
bash scratch/quick-search.sh --dry-run
```

Launch the six configurations with:

```shell
bash scratch/quick-search.sh
```
