# Inputs and target registration

## Target constraint

The controller reads one JSON object with the `TargetConstraint` fields defined
in `trex/schemas.py`. A minimal example is:

```json
{
  "target_id": "example_target_v1",
  "target_class": "enzyme",
  "hotspots": ["A42", "A77"],
  "chain_ids": ["A"],
  "panel_size_K": 8
}
```

`target_id` must be stable within one archive. `chain_ids` identify the target
chains retained in the input complex. Each hotspot is a PDB chain ID followed
by its residue number and optional insertion code.

Optional fields are:

- `forbidden_surfaces`
- `assay_geometry`
- `developability_filters`

They are structured context for planning; they do not alter the fixed strict
success thresholds.

## Structure requirements

- Use PDB format for the registered target input.
- Preserve the intended chain IDs and residue numbering.
- Remove unintended binders, ligands, or chains before registration.
- Ensure every hotspot exists in a configured target chain.
- Record the exact PDB SHA256.

The validator parses the first model, verifies all chains and hotspots, and
checks registered assets against `config/targets/assets.sha256.json`.

## Registered targets

`config/targets/registry.json` maps the public target key used by launchers to:

- a target-constraint JSON;
- an expected target ID; and
- a path relative to `TREX_TARGET_ASSET_ROOT`.

Do not edit a registered constraint or replace its PDB while reusing the same
key. Add a new key and update the asset manifest.

## Custom targets

Custom targets need not be added to the paper registry:

```bash
trex-validate \
  --target my_target \
  --target-config /path/to/my_target.json \
  --target-pdb /path/to/my_target.pdb \
  --enabled-families "$TREX_ENABLED_FAMILIES" \
  --require-backends
```

The custom structure, chains, and hotspots are validated, but no registered
SHA256 comparison is claimed. The resulting `run_provenance.json` records the
actual config and PDB hashes.

Use `trex-target init` to create the constraint and `trex-target resolve` to
inspect the exact target ID/config/PDB triple a launcher will use. The local and
Slurm launchers accept `TREX_TARGET_CONFIG` plus `TREX_TARGET_PDB` for an
unregistered target; registered labels resolve through `registry.json` and
`TREX_TARGET_ASSET_ROOT`.

## Generated artifact formats

T-ReX does not ignore every `.cif` file. Admission is parser- and
route-specific: a generated artifact must have a validated structure/sequence
extraction and canonical scoring path. Unsupported or incomplete artifacts
remain archived as diagnostic/failed outputs and cannot mint strict or SU
credit.
