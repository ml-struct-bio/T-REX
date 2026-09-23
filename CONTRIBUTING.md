# Contributing

Open an issue before changing controller policy, qualification/SU criteria,
archive schemas or backend contracts. Keep pull requests narrowly scoped and
include tests for behavioral changes.

From the repository root, using Python 3.10–3.12:

```bash
python -m pip install -e '.[dev]'
python -m pip check
bash scripts/release_check.sh python
```

The release check runs the regression suite, deterministic policy checks,
resource/prompt checks, shell syntax checks and an
installed-wheel smoke outside the checkout. It requires Git and a package index
or populated cache. These CPU checks use synthetic inputs and mocked external
processes; they do not establish molecular-campaign throughput.

Preserve archive keys, action-family IDs, CLI flags, defaults and LLM prompt
bytes unless intentionally changing their contracts.

Do not commit model weights, scientific target structures, generated designs,
credentials, private paths, logs or campaign archives. The analysis example's
synthetic structure is a test fixture. See [outputs and analysis](docs/outputs-and-analysis.md)
for the files and provenance retained with a run.

## Add a backend

The existing adapter interface supports new generators and sequence-redesign
methods. Start with a skeleton; it is not a working integration until its
command builder and output parser match your tool:

```bash
trex-backend init --family my_generator \
  --executable /absolute/path/to/my-generator \
  --out-dir ./external/my_generator_adapter
```

Edit `trex_my_generator_backend.py` in that directory:

- `build_command`: construct the executable arguments and environment from the
  supplied target, candidate settings and optional parent structure.
- `parse_output`: validate the actual output files and return `ResultRecord`
  entries with unique IDs, target/parent provenance and accessible structure paths.
- `Capability` and budget fields: define the role, permitted parameters,
  executable/checkpoint checks, workload limits and timeout. A redesign adapter
  normally needs `role="seq_redesign"` and `requires_parent_pdb=True`.

Use a new family name outside the reserved `complexa_` prefix. Native scores
must use backend-prefixed names and `outputs_diagnostic_only=True`; the adapter
cannot supply the canonical qualification fields directly. Keep
`structure_refilter` enabled so eligible generated structures can receive
standardized AF2 evaluation before qualification and SU credit. A new scorer
or qualification definition requires a core evaluation change.

For Slurm, install the adapter into the environment selected by
`TREX_CONTROLLER_PYTHON`. Add this `pyproject.toml` beside the generated module:

```toml
[build-system]
requires = ["setuptools>=77"]
build-backend = "setuptools.build_meta"

[project]
name = "trex-my-generator-adapter"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["trex-binder-design>=0.1.2"]

[project.entry-points."trex.backends"]
my_generator = "trex_my_generator_backend:ADAPTER"

[tool.setuptools]
py-modules = ["trex_my_generator_backend"]
```

After implementing the adapter and loading `.env`:

```bash
"$TREX_CONTROLLER_PYTHON" -m pip install -e ./external/my_generator_adapter
"$TREX_CONTROLLER_PYTHON" -m trex.scaffold list
"$TREX_CONTROLLER_PYTHON" -m trex.scaffold validate
```

Append `my_generator` to `TREX_ENABLED_FAMILIES` in `.env`, or to
`run.enabled_families` in YAML, and run the full target/backend preflight from
the [run guide](docs/slurm.md#validate-and-submit). Test command creation,
output parsing and AF2 handoff before a full campaign. Passing registration
and preflight alone does not test molecular inference.

For local development, an importable module can instead be loaded through
`TREX_BACKEND_PLUGINS=trex_my_generator_backend:ADAPTER`. Use one registration
method: registering the same family both ways is rejected. The Slurm wrapper
sets `PYTHONPATH` to the T-REX checkout, so relying only on an extra external
`PYTHONPATH` entry will not work there; use an installed adapter.

Registration exposes the family's declared parameters to planning and connects
its command/parser to execution. It does not add a mandatory initial trial:
the initial seed-job list remains specific to the built-in families. Existing
prompts and deterministic policies also retain guidance for those methods;
new family-specific scheduling or reasoning rules require separate changes.
