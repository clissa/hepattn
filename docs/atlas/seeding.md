# ATLAS Seeding And Reproducibility

This note summarizes how random seeding is wired for the ATLAS workflow in this
checkout, how to change the configured seed, and what that does and does not
guarantee for reproducibility.

It is based on:

- `src/hepattn/experiments/atlas/main.py`
- `src/hepattn/experiments/atlas/configs/base.yaml`
- `src/hepattn/experiments/atlas/configs/base_zj1234.yaml`
- `src/hepattn/experiments/atlas/configs/base_inference.yaml`
- `src/hepattn/experiments/atlas/pflow_data.py`
- `src/hepattn/utils/cli.py`
- the installed Lightning source under `.hepattn/lib/python3.12/site-packages`

## Entry Point

ATLAS training starts from:

```shell
cd src/hepattn/experiments/atlas
python main.py fit -c configs/base.yaml
```

`main.py` wires the repo's custom Lightning CLI to:

- model class: `hepattn.experiments.atlas.lightning_module.MPflow`
- data module: `hepattn.experiments.atlas.pflow_data.PflowDataModule`
- default fit config: `configs/base.yaml`

## Lightning Seed

The repo CLI subclasses LightningCLI in `src/hepattn/utils/cli.py`. The seed
argument itself is provided by LightningCLI as:

```text
--seed_everything
```

LightningCLI applies this seed before model and datamodule instantiation. In
the installed Lightning version in this checkout, `seed_everything` sets:

- Python's `random` module
- NumPy's global random state
- Torch's global random state
- `PL_GLOBAL_SEED`
- `PL_SEED_WORKERS`, when called with `workers=True`

The checked-in ATLAS configs set:

```yaml
# configs/base.yaml
seed_everything: 20260616
```

```yaml
# configs/base_zj1234.yaml
seed_everything: 20260619
```

```yaml
# configs/base_inference.yaml
seed_everything: 42
```

`docs/training.md` also identifies `seed_everything` as the global training
seed.

## Changing The Training Seed

For a one-off run, override the config value from the command line:

```shell
cd src/hepattn/experiments/atlas

python main.py fit \
  -c configs/base.yaml \
  --seed_everything=12345
```

For a reusable experiment config, edit or copy the YAML and change the
top-level value:

```yaml
seed_everything: 12345
```

This controls the Lightning-level seed applied before the model and datamodule
classes are instantiated. It should affect model initialization and other RNG
use that happens before the ATLAS dataset constructor resets the seed.

## Dataset Seed

The ATLAS dataset has a second seed path that is not currently configurable
from YAML:

```python
self.sampling_seed = 42
np.random.default_rng(self.sampling_seed)
seed_everything(self.sampling_seed, workers=True)
```

This runs inside `ATLASDataset.__init__` in
`src/hepattn/experiments/atlas/pflow_data.py`. Because datasets are constructed
during datamodule setup, this call happens after LightningCLI has already
applied the top-level `seed_everything` value.

The practical consequence is that changing `--seed_everything` changes the
initial Lightning/model seed, but it is not a single source of truth for the
whole ATLAS training run. Dataset construction resets the global Python,
NumPy, Torch, and Lightning worker seed state back to `42`.

The ATLAS train dataloader also uses:

```python
shuffle=True
```

without passing an explicit `torch.Generator`. Therefore the training sample
order is tied to the global Torch RNG state at dataloader construction time.

To make dataset seeding configurable, the ATLAS datamodule or dataset would
need to expose a `sampling_seed` argument in config and avoid hard-coding `42`
inside the dataset constructor.

## Current Reproducibility Status

The ATLAS workflow is seeded, but it should not be described as fully
reproducible in the strict bitwise sense.

The main limitations are:

- `seed_everything` is not a single source of truth because `ATLASDataset`
  resets global RNG state to `42` during dataset construction.
- The train dataloader shuffles samples without an explicit generator.
- The checked-in ATLAS configs do not set `trainer.deterministic`.
- The code does not call `torch.use_deterministic_algorithms`.
- There are no explicit cuDNN deterministic or benchmark settings.
- The default ATLAS configs use GPU training, multi-device execution,
  `bf16-mixed` precision, and `flash-varlen` attention, none of which is by
  itself a guarantee of exact repeatability.
- Some model utilities use compiled paths, and other experiment configs enable
  the `hepattn.callbacks.Compile` callback. Compiled kernels can affect exact
  numerical repeatability across software and hardware environments.

The current setup is sufficient for controlled single-seed experiments where
the seed, config, data files, hardware, software stack, and checkpoint are
recorded. It is not enough to claim exact GPU-training reproducibility unless
that was checked with repeated runs under fixed conditions.

## Practical Checklist

For an ATLAS run where reproducibility matters, record:

- git revision or patch context
- full resolved `config.yaml`
- `seed_everything`
- whether the dataset seed is still the hard-coded `42`
- dataset paths and split definitions
- `num_train`, `num_val`, `num_test`, and filtering limits
- `trainer.devices`, accelerator, precision, and deterministic settings
- attention backend, especially `flash-varlen` versus `torch`
- Torch, Lightning, CUDA, and FlashAttention versions
- hardware and GPU IDs
- checkpoint path and evaluation command

For stronger repeatability, consider changing the implementation to:

- expose `data.sampling_seed` in the ATLAS config
- avoid resetting global RNG state inside `ATLASDataset.__init__`
- pass an explicit generator to the train dataloader when shuffling
- set and document Lightning/PyTorch deterministic options where compatible
- compare at least two repeated runs on the same machine before claiming exact
  reproducibility

## Verification

This document is a source audit. It does not claim that a repeated ATLAS
training run was executed or that GPU determinism was verified.
