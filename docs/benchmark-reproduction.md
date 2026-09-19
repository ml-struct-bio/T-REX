# Benchmark reproduction

## Numerical audit included in Git

`benchmarks/paper_7target_144h/` contains the frozen source tables for the
primary seven-target, six-method endpoint comparison. The manifest fixes:

- the seven targets and six methods;
- 144 H100 worker GPU-hours per target-method arm;
- the three-filter logical-AND gate;
- binder-chain Foldseek TM0.60 clustering; and
- SHA256 for every source table.

```bash
python benchmarks/paper_7target_144h/verify.py
```

This verifies the complete seven-target endpoint, Foldseek-sensitivity and
MMseqs2 tables and recomputes the reported geometric-mean T-ReX/PUCT and
T-ReX/target-wise-best-fixed ratios from `endpoints_tm06.csv`.

## Re-running campaigns

To reproduce the protocol:

1. check out this T-ReX release;
2. install the revisions in `production_stack.json`;
3. stage the exact target structures and Qwen model matching their manifests;
4. use three H100 worker slots and the same 144 worker-GPU-hour cutoff;
5. keep the strict and Foldseek thresholds unchanged;
6. enable the same action families;
7. retain `run_provenance.json` and every append-only stream; and
8. recompute endpoints from all eligible artifacts at the common cutoff.

The reference Slurm layout reserves a fourth GPU for the LLM. That GPU is
excluded from the primary worker-GPU-hour denominator.

## What exact reproduction means

The following should reproduce exactly:

- source tree and asset/model digests;
- target definitions, action registry, filters, clustering thresholds, and
  accounting rules;
- deterministic unit/replay outputs; and
- calculations from the frozen source tables.

The following are not expected to be bitwise identical:

- generated sequences and structures;
- asynchronous completion order;
- LLM text;
- GPU kernel reductions; and
- the final number of SU in a newly sampled campaign.

Thus a new run reproduces the experimental protocol and should be interpreted
as another stochastic campaign, not a replay of identical molecular identities.

## Raw archive release contract

A complete external raw-data deposit must contain, per target-method arm:

- every `run_provenance*.json` and `repository_patches/`;
- every JSONL controller stream;
- `controller_checkpoint.json`;
- generated structures and canonical score files;
- Foldseek/MMseqs2 inputs, outputs, and command/version metadata;
- target/config/model content hashes;
- event-time provenance and the 144 worker-GPU-hour eligibility cutoff; and
- a top-level SHA256 manifest.

**Archive DOI: pending public deposition.**

Until that DOI is filled, the numerical endpoint/source-table audit is public
and deterministic, but an external user cannot reconstruct every post-hoc
molecular analysis from raw structures. The README labels this boundary
explicitly.
