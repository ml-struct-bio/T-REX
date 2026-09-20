# CPU archive-analysis example

From the repository root, after `python -m pip install -e .`:

```bash
python examples/analysis_demo/run_demo.py --out ./demo-output
trex-analyze validate --archive-root ./demo-output/archive
trex-analyze summary --archive-root ./demo-output/archive
trex-analyze trace --archive-root ./demo-output/archive --limit 10
```

Choose a new output directory for each run. Existing files are preserved.
The example uses the installed package and existing analysis/export commands;
it requires no GPU, model weights, LLM server, clustering tools or test suite.

## Inputs

[fixture.json](fixture.json) supplies two artificial measurement records.
[toy.pdb](toy.pdb) contains one synthetic coordinate for testing file copying.
The script creates one linked hypothesis, candidate job, selection, dispatch and
evidence summary. No worker runs and no molecular metric is computed. A
synthetic `started` record demonstrates the archive format only.

## Outputs

```text
demo-output/
  archive/          JSONL records, DEMO_ONLY.json and toy worker output
  analysis/         validate.json, summary.json, trace.json, schema.json, export.json
  export/           manifest.csv, manifest.json and pdbs/
  checks.json       observed values compared with the expected contract
```

[Expected checks](expected_outputs/checks.json): two result records, one fixture
meeting qualification cutoffs, one decision-trace entry and one exported PDB.
The unqualified fixture is excluded from the export. The CSV contains result IDs,
measurements, ranking and source/destination paths. Absolute artifact paths are
created for your chosen output directory.

Validation succeeds with warnings about unperformed clustering and absent
production provenance. They remain visible in `analysis/validate.json`.
No provenance is fabricated for an experiment that did not occur.

Foldseek status is `not_run`; throughput is null. The raw archive's integer
`run_su_count=0` is an uncomputed fixture placeholder, **not a measured SU count**.
The export uses `--no-dedup` to demonstrate qualification/ranking/file copying
without clustering binaries. Its bin annotations do not establish structural
uniqueness. For real data, follow the [analysis guide](../../docs/outputs-and-analysis.md).

For a separate export of the same fixture:

```bash
trex-export --archive-root ./demo-output/archive --target-id synthetic_demo \
  --n 10 --no-dedup --out-dir ./demo-export
```
