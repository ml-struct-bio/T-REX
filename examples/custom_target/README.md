# Run a new target

A custom target does not need an entry in the built-in registry. Prepare a PDB
and a copy of [target.json](target.json), replacing its target ID, class, chain
IDs and hotspots. Keep the target ID stable within one campaign.

`chain_ids` identifies target chains in the input PDB. Each hotspot is a chain
ID followed by its residue number and optional insertion code; it must exist
in a configured target chain. Validation checks these constraints and records
the actual PDB/JSON hashes. A custom input has no registered hash to compare to.

## Configure the campaign

Keep installed backend/model paths in the repository-root `.env`, then create a
campaign YAML with the explicit custom inputs:

```bash
trex init campaign.yaml \
  --target my_target \
  --target-constraint /absolute/path/to/my_target.json \
  --target-pdb /absolute/path/to/my_target.pdb \
  --gpus 4 \
  --hours 48 \
  --output /absolute/path/to/my-target-runs
```

Set `run.enabled_families` in `campaign.yaml` to
`bindcraft`, `boltzgen`, `proteinmpnn_redesign` and `structure_refilter` when
Complexa has no matching target entry. Then validate and submit:

```bash
trex check campaign.yaml
trex submit campaign.yaml
```

Use a new output directory for an independent campaign. `trex export` infers the
custom target ID from the recorded campaign input.

## Include Complexa generation

The current Complexa adapter passes the JSON's `target_id` as
`generation.task_name`. It does not translate T-REX's PDB, chains and hotspots
into a new Complexa target definition. In your Complexa checkout, add a matching
entry to `configs/targets/targets_dict.yaml` using its target-management tools
(`complexa target add -i` in the activated Complexa environment). Use exactly
the same target ID, structure, target chains and hotspot residues.

Then add the desired `complexa_*` families to the enabled list. T-REX's current
preflight does not validate that Complexa target entry; check it in Complexa
before launching. Preserve the backend configuration changes in run provenance.
If only the T-REX PDB path is changed, Complexa can still resolve a different
structure through its own configuration.

For targets outside the built-in length table, the current controller supplies
an 80–150-residue binder-length range to BindCraft and BoltzGen. The target JSON
does not expose this range; changing it requires a controller change.
Complexa uses the binder-length range in its own target entry.
