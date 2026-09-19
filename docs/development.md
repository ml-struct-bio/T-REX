# Development and verification

Run these commands from the repository root with Python 3.10, 3.11 or 3.12:

```bash
python -m pip install -e '.[dev]'
python -m pip check
bash scripts/release_check.sh python
```

The release check runs the regression suite, deterministic policy checks,
publication resource and prompt checks, frozen-table verification, shell syntax
checks, and an installed-wheel smoke outside the source checkout. It requires
a Git checkout and a working package index or populated package cache.
CI runs this gate on all three Python versions.

The tests cover candidate-job construction and selection, campaign-state
classification, archive contracts, accounting, prompts, execution, resume and
shutdown. CPU checks use synthetic inputs and mocked external processes.
They do not run molecular inference or establish campaign throughput.

## Independent wheel installation


From the repository root with `.[dev]` installed, build a wheel and install it
with its declared dependencies into a separate virtual environment. This also
checks all nine public command-line entry points and their bounded workflows
outside the source checkout, without GPUs, model weights, or backend installs.

```bash
TREX_WHEEL_CHECK_ROOT="$(mktemp -d)"
TREX_SOURCE_CHECKOUT="$PWD"
python -m build --outdir "$TREX_WHEEL_CHECK_ROOT/dist"
python -m venv "$TREX_WHEEL_CHECK_ROOT/venv"
"$TREX_WHEEL_CHECK_ROOT/venv/bin/python" -m pip install \
  "$TREX_WHEEL_CHECK_ROOT"/dist/*.whl
"$TREX_WHEEL_CHECK_ROOT/venv/bin/python" -m pip check
(
  cd "$TREX_WHEEL_CHECK_ROOT"
  PYTHONPATH= "$TREX_WHEEL_CHECK_ROOT/venv/bin/python" \
    "$TREX_SOURCE_CHECKOUT/scripts/installed_wheel_smoke.py" \
    --bin-dir "$TREX_WHEEL_CHECK_ROOT/venv/bin" \
    --expected-prefix "$TREX_WHEEL_CHECK_ROOT/venv"
)
```

This check resolves dependencies independently. The standard
`scripts/release_check.sh` additionally runs the full test suite, deterministic
policy checks and frozen benchmark audit; its wheel smoke reuses dependencies
from the test environment. Neither check executes a scientific backend.



## GPU allocation smoke

The [GPU smoke template](../slurm/multigpu_smoke.slurm) checks distinct visible
worker devices and bounded controller/dispatch behavior inside a Slurm
allocation. Configure the account, GPU resources, shared-storage Python
environment and `TREX_REPO_ROOT` for your cluster. It does not run molecular
inference. See [installation](installation.md) for backend validation.

## Code and terminology

See [architecture](architecture.md), the [manuscript-to-code glossary](terminology.md)
and [backend extension contracts](extending.md). Archive keys, action-family IDs,
CLI flags, defaults and LLM prompt bytes are compatibility contracts. Preserve
them when improving internal names. Keep regression fixtures under `trex/tests/`;
tests and development checks are excluded from the installed runtime package.
