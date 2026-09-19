# Contributing

Please open an issue before changing controller policy, success criteria, archive
schemas, or backend contracts. Pull requests should be narrowly scoped and must
include tests for behavioral changes.

Run before submitting:

```bash
./scripts/release_check.sh
```

Do not commit scientific target structures, model weights, generated designs, credentials,
cluster logs, private paths, or run archives.

New scientific methods should use the diagnostic-only adapter contract in
`docs/extending.md`. A new canonical scorer or a change to success/SU semantics
is a core evaluation change, not a backend plugin.

The one-coordinate synthetic PDB in `examples/analysis_demo/` is a software-test
fixture. See [development and verification](docs/development.md) for installation
and release checks, and [terminology](docs/terminology.md) for manuscript names.
