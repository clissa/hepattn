# ATLAS MDN quick-search round 2

This round repeats the strongest round-1 directions with the complete JZ1--4
training file, a larger per-device batch, and a longer epoch budget.

| Config | Round-1 parent | Additional feature |
| --- | --- | --- |
| `qs13_ref_detw60` | `qs00_ref_detw60` | reference |
| `qs14_null02` | `qs07_null02` | classification null weight 0.2 |
| `qs15_incaux` | `qs06_incaux` | intermediate incidence supervision |
| `qs16_detw30` | `qs11_detw30` | deterministic loss weight 30 |
| `qs17_matchreg3` | `qs08_matchreg3` | regression matching cost 3 |
| `qs18_null02_incaux` | `qs07_null02` | null weight 0.2 plus intermediate incidence supervision |

Every config uses `JZ1234_train.root`, `batch_size: 112`, two devices,
`accumulate_grad_batches: 2`, `num_train_per_epoch: 12000`, and 30 epochs.
All other settings are inherited from the named round-1 parent.

## Training-budget comparison

`use_distributed_sampler: false` means each of the two ranks independently
draws 12,000 training events per epoch. The total processed examples below are
nominal: the two rank-local samples can overlap.

| Round | Batch/rank | Batches/rank/epoch | Optimizer steps/epoch | Epochs | Optimizer steps | Processed examples/rank | Nominal total processed examples |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 80 | 150 | 75 | 22 | 1,650 | 264,000 | 528,000 |
| 2 | 112 | 108 | 54 | 30 | 1,620 | 360,000 | 720,000 |

Round 2 therefore processes 36.4% more examples while taking 1.8% fewer
optimizer steps. Its effective global batch per optimizer step rises from 320
to 448 examples.

## Launch

From the repository root, inspect the planned two-wave launch with:

```shell
bash scratch/quick-search.sh --dry-run
```

Launch the six configurations with:

```shell
bash scratch/quick-search.sh
```
