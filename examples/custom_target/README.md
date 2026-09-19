# Custom target template

1. Copy `target.json` and replace the target ID, class, chain IDs, and hotspots.
2. Prepare a PDB whose chain IDs and residue numbering match the JSON.
3. Validate before any GPU work:

```bash
trex-validate \
  --target my_target \
  --target-config /path/to/my_target.json \
  --target-pdb /path/to/my_target.pdb \
  --enabled-families "$TREX_ENABLED_FAMILIES" \
  --require-backends
```

This template is illustrative and is not a benchmark target. See
`docs/inputs.md` for the input contract.
