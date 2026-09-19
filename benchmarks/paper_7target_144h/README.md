# Frozen seven-target benchmark source data

This directory contains the frozen, hash-verified source tables for the reported
six-method, seven-target comparison at a common cutoff of 144 H100 campaign-worker
GPU-hours per method--target campaign. Manuscript throughput rates divide the
endpoint counts by this fixed nominal 144-hour denominator; measured worker
exposure is retained separately as audit metadata. These are the numerical
source tables used by the manuscript's 2026-08-05 freeze.

Run the audit from the repository root:

```bash
python benchmarks/paper_7target_144h/verify.py
```

The verifier checks file hashes, the exact seven-target by six-method endpoint
matrix, cross-table TM0.6 counts, the TM0.5/TM0.6/TM0.8 sensitivity grid,
MMseqs2 sequence-cluster results and the two headline geometric-mean ratios.
It also verifies the reported total of 1,461 T-ReX TM0.6 structure-unique units,
T-ReX's lead in all 21 target-by-Foldseek-threshold comparisons, and its
sequence-cluster-throughput lead on all seven targets.

Files:

- `endpoints_tm06.csv`: primary endpoint counts and worker exposure;
- `curves_tm06.csv`: audited TM0.6 worker-exposure trajectories;
- `tm_sensitivity.csv`: Foldseek TM0.5, TM0.6 and TM0.8 endpoint counts;
- `mmseqs70_clusters.csv`: 70%-identity, 0.8-coverage MMseqs2 analysis;
- `manifest.json`: scope, expected headline values and SHA-256 hashes; and
- `verify.py`: dependency-free numerical and integrity audit.

Some `data_status` values retain names such as `six_target` because they are
immutable provenance labels from the source audits in which those individual
runs were first frozen. They are not the scope of this package: the verifier
requires all seven registered targets, including IL7RA, in every result table.

These compact tables do not replace the raw structure and campaign archives.
New campaigns are stochastic and are expected to reproduce the declared
protocol and statistical behavior, not exact molecular identities or counts.
