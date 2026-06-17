# Installation

This repo is configured for Pixi, but a plain Python virtual environment can
also work when starting from a container that already provides the CUDA driver
stack and, optionally, a compatible PyTorch build.

## Recommended Baseline

The most reproducible setup is still the Pixi environment from the README:

```bash
pixi install --locked
pixi shell
```

That path uses the lock file and the dependency choices encoded in
`pyproject.toml`, including Python 3.12, CUDA channels, PyTorch, and
FlashAttention.

Use the venv instructions below when Pixi is not available.

## Fresh Venv With Repo-Like Torch

Use this when you want a clean environment that is close to the repo's expected dependency stack.

```bash
cd <repo_basepath>/hepattn

python3.12 -m venv .hepattn
source .hepattn/bin/activate

python -m pip install -U pip setuptools wheel
python -m pip install packaging ninja scikit-build-core pybind11

python -m pip install torch==2.7.0 \
  --index-url https://download.pytorch.org/whl/test/cu126

python -m pip install -e . \
  --no-build-isolation \
  --extra-index-url https://download.pytorch.org/whl/test/cu126
```

`--no-build-isolation` is important because the project config uses it for
`flash-attn`. It also means the build backend must already be installed in the
venv, hence the explicit `scikit-build-core` and `pybind11` install.

d
After installing, you may need to add `export TRITON_PTXAS_PATH=/<path-to-venv>/.hepattn/lib/python3.12/site-packages/triton/backends/nvidia/bin/ptxas`
to your `.bashrc` or equivalent to ensure the Triton compiler can find `ptxas` at runtime.

<!-- ## Venv Reusing Container Torch

Use this when the container already provides the exact PyTorch build you want
to keep, for example an NVIDIA container build such as:

```text
torch 2.10.0a0+a36e1d39eb.nv26.01.42222806
```

In that case, prefer a venv with system site packages and install `hepattn`
without dependencies, so pip does not replace or fight the container's Torch
stack.

```bash
cd <repo_basepath>/hepattn

python3.12 -m venv --system-site-packages .hepattn
source .hepattn/bin/activate

python -m pip install -U pip setuptools wheel
python -m pip install packaging ninja scikit-build-core pybind11

python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"

python -m pip install -e . --no-deps --no-build-isolation
```

Then install any missing runtime dependencies manually. Avoid installing
`hepattn` with dependencies in this mode unless you are willing to let pip
change the container's Python package stack. -->

## FlashAttention Caveat

`pyproject.toml` currently declares a direct `flash-attn` wheel:

```text
flash_attn-2.7.4.post1+cu12torch2.6...cp312...
```

That wheel is compiled for a specific CUDA/PyTorch ABI. It is much more likely
to work in the repo-like Torch setup than with a newer NVIDIA alpha PyTorch
build.

The code also imports `flash_attn` at module import time in
`src/hepattn/models/attention.py`. If `flash-attn` is missing or binary
incompatible, importing attention/model modules may fail even when a config
uses `attn_type: torch`.

For development against a container Torch build, the safest options are:

- use configs with `attn_type: torch` or `attn_type: flex`
- make `flash_attn` optional in the code before relying on non-flash backends
- build/install a `flash-attn` version that is compatible with the container's
  PyTorch build
<!-- 
## Solver Caveat

The bundled `lap1015` package is a compiled solver extension. In some
container/venv combinations it can import but later segfault during direct
solver tests.

Most training configs use the SciPy matcher. Prefer:

```yaml
default_solver: scipy
adaptive_solver: false
```

The direct `lap1015` stress test is:

```bash
pytest tests/matching/test_solvers.py
```

If this segfaults, do not treat it as proof that the Torch/GPU setup is broken.
It means the native `lap1015` extension is not trustworthy in that environment.
 -->
## Checks

Basic import and GPU checks:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
python -c "import hepattn; print('hepattn import OK')"
```

If you are using model modules, also check:

```bash
python -c "from hepattn.models.attention import Attention; print('attention import OK')"
```

<!-- Run tests that do not require GPU or external data:

```bash
pytest -m 'not gpu and not requiresdata'
```

Some experiment tests currently access paths such as `data/tide/raw`,
`data/tide/prepped`, `data/trackml/raw`, and `data/trackml/prepped` without all
of them being marked `requiresdata`. Missing local data can therefore produce
failures in a minimal environment.

When validating the Python package while avoiding the direct `lap1015` crash,
use:

```bash
pytest -m 'not gpu and not requiresdata' --ignore=tests/matching/test_solvers.py
```

or run a focused subset:

```bash
pytest tests/loss tests/matching/test_matcher.py tests/matching/test_match_equivalence.py tests/flex tests/models \
  -k 'not lap1015'
``` -->

## Notes On Existing Container Environments

If `pip check` reports conflicts from packages such as `mlxtend` or `relbench`,
those are not dependencies used by this repo. They can conflict with
`scikit-learn` constraints in the wider container environment, but they are not
required by `hepattn`.

The repo itself depends on `scikit-learn>=1.6.1,<2`, mainly for evaluation and
matching utilities.

For reliable development, avoid installing into a broad, already-populated
container Python environment. Use either a clean venv or a
`--system-site-packages` venv only when you intentionally want to reuse the
container's Torch/CUDA Python packages.
