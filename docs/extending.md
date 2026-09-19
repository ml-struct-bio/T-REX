# Extending T-ReX

T-ReX has two extension surfaces: target inputs and scientific backends. Both
are validated before a GPU worker starts. Extensions enlarge the action space;
they do not change the canonical success gate or Foldseek SU definition.

## Add a target or hotspot set

Create a constraint template:

```bash
trex-target init \
  --out /path/to/egfr.json \
  --target-id egfr_crop_v1 \
  --target-class receptor \
  --chain A \
  --hotspot A123 \
  --hotspot A167
```

Then validate the exact PDB, chain IDs, and residue numbering:

```bash
trex-validate \
  --target egfr \
  --target-config /path/to/egfr.json \
  --target-pdb /path/to/egfr.pdb \
  --enabled-families "$TREX_ENABLED_FAMILIES" \
  --require-backends
```

The Slurm and local launchers accept the same custom pair:

```bash
export TARGET=egfr
export TREX_TARGET_CONFIG=/path/to/egfr.json
export TREX_TARGET_PDB=/path/to/egfr.pdb
sbatch --export=ALL slurm/trex_per_target_node.slurm
```

For a shared registered target, add one new entry to
`config/targets/registry.json`, put its constraint beside the registry, and add
the exact PDB digest to `assets.sha256.json`. The launchers read the registry at
runtime; no shell target table needs editing. Use a new target key and target ID
when changing a crop, repair, chain assignment, or hotspot definition.

## Add a generator or redesign backend

Generate a working adapter skeleton:

```bash
trex-backend init \
  --family my_generator \
  --executable my-generator \
  --out-dir extensions/my_generator
```

The generated adapter has four explicit responsibilities:

1. declare its role, cost class, parent requirement, and allowed parameter
   ranges in a `Capability`;
2. declare the multiplicative work-unit parameters, defaults, and per-launch
   cap used by the deterministic budget guard;
3. convert a validated `BackendLaunchContext` into a direct argv
   `BackendCommand`; and
4. parse backend artifacts into immutable `ResultRecord` rows.

T-ReX does not execute a shell string from the Planner. The LLM can select only
the registered family and parameter values inside the declared ranges. The
controller owns GPU assignment, subprocess isolation, output containment,
timeouts, cost attribution, and append-only archiving.

During local development, load the adapter with an explicit module reference:

```bash
export PYTHONPATH="$PWD/extensions/my_generator:$PYTHONPATH"
export TREX_BACKEND_PLUGINS=trex_my_generator_backend:ADAPTER
export TREX_ENABLED_FAMILIES="$TREX_ENABLED_FAMILIES,my_generator"

trex-backend list
trex-backend validate
trex-validate --target cd45 \
  --asset-root "$TREX_TARGET_ASSET_ROOT" \
  --enabled-families "$TREX_ENABLED_FAMILIES" \
  --require-backends
```

A distributable Python package should publish the adapter through the entry
point group `trex.backends`:

```toml
[project.entry-points."trex.backends"]
my_generator = "my_package.trex_adapter:ADAPTER"
```

Set `TREX_DISABLE_ENTRYPOINT_BACKENDS=1` to disable automatic discovery. An
explicit comma-separated `TREX_BACKEND_PLUGINS=module:attribute,...` remains
available for source-checkout development.

## Score isolation

Operator-installed backends are intentionally limited to `generator` or
`seq_redesign` roles and must set `outputs_diagnostic_only=True`. Their native
confidence, interface, or energy values must use backend-prefixed metric names.
The adapter validator rejects canonical `pLDDT`, `iPAE`, or
`binder_scRMSD` fields from extension records. Generated structures enter the
existing `structure_refilter` path, and only that canonical scoring result can
pass the three-filter gate and receive Foldseek SU credit.

A new canonical scorer is not a drop-in adapter. It changes the evaluation
contract and requires a reviewed core integration, calibration evidence, and a
separate experiment.

## Configuration ranges

`Capability.allowed_params` is the only Planner-visible configuration surface:

```python
allowed_params={
    "num_designs": (1.0, 64.0),
    "temperature": (0.05, 2.0),
    "protocol": ["default", "interface-focused"],
}
```

Unknown categorical values are dropped; numeric overshoots are clamped to the
declared range. Budget-driving parameters must also appear in
`budget_parameters` and `budget_defaults`. Their product is checked against
`eval_budget_cap` before dispatch. Do not widen the frozen ranges of a reported
built-in campaign and call it the same experiment; record the change as a new
configuration or adapter revision.

## Release checklist for an extension

Before a full campaign:

1. test command construction without a GPU;
2. test the parser on successful, partial, empty, malformed, and resumed output;
3. run `trex-backend validate` and `trex-validate --require-backends`;
4. run the complete T-ReX unit and deterministic replay suite;
5. run a bounded compute-node campaign smoke;
6. verify archive lineage, GPU-hour attribution, score-conversion coverage, and
   Foldseek status; and
7. archive the adapter source hash and backend revision. T-ReX writes loaded
   adapter source hashes and budget metadata into `run_provenance.json`.
