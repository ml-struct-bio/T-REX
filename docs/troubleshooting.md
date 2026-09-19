# Troubleshooting

## Preflight reports a backend that I disabled

Pass the exact comma-separated family list to both preflight and the controller:

```bash
export TREX_ENABLED_FAMILIES=complexa_beam,complexa_best_of_n,structure_refilter
```

The current preflight is conditional. An unconditional BindCraft or BoltzGen
failure indicates an older launcher.

## Foldseek is installed but not found

Prefer:

```bash
export TREX_FOLDSEEK_BIN=/absolute/path/to/foldseek
```

The Slurm wrapper prepends its directory to `PATH`. On module-based systems set
`TREX_FOLDSEEK_MODULE` instead. Do not load a cluster CUDA toolkit into every
backend environment manually.

## Target hash fails

The file is not the frozen benchmark crop/repack. Check:

- the relative file selected by `registry.json`;
- PDB chain IDs and residue numbering;
- whether a preparation step rewrote coordinates; and
- the expected SHA in `assets.sha256.json`.
Register a deliberately changed structure under a new target key.

## Backend revision fails after applying the BindCraft patch

Revision validation checks both `git rev-parse HEAD` and the expected tracked
BindCraft diff. A clean tree or a different local patch fails the publication
preflight even when `HEAD` is correct. Run provenance records the same commit
and tracked-diff hash and stores a patch artifact beside the archive.

## Model verification is slow or fails

Startup validates manifest integrity, file names, and byte sizes against the
checked-in full-content manifest; it does not rehash the large weight tree at
every launch. A missing file or changed size is fatal. Use
`trex-provenance hash-model` once to create a manifest for a new model.

For a byte-level release audit, add `--verify-checkpoint-content` to
`trex-validate` or add `--full-content` to `trex-provenance model-digest`. The
publication preflight enables this slower mode.

## SU is zero despite strict successes

Check:

```bash
trex-analyze summary --archive-root /path/to/archive
trex-analyze validate --archive-root /path/to/archive
```

Strict records cannot mint SU when Foldseek is unavailable, failed, has
incomplete coverage, or did not cluster the binder chain. Native backend
acceptance is not a substitute.

## Many outputs are pending canonical scoring

Diagnostic generators can produce batches faster than canonical AF2 scoring.
The controller uses progressive score conversion and records
`diagnostic_chain_backlog`, refilter roles, and dispatch realization in each
EvidenceSummary. Inspect the trace before concluding that the generator itself
is dry. Do not manually mark native accepted files as strict.

## Selected work did not start

`LaunchDecision` is selection intent; `DispatchRecord` is actual execution.
Inspect:

```bash
trex-analyze trace --archive-root /path/to/archive --limit 20
```

Common causes are a busy worker slot, stale prefetched action, parent artifact
loss, route cap, or dispatch failure. The archive preserves the reason.

## Resume appears to restart accounting

Use `TREX_RESUME_ARCHIVE` with the Slurm wrapper. A resume writes a new
provenance file and continues the append-only archive. `TREX_MAX_WALL_H` is the
total campaign wall limit, including the elapsed offset recovered from the
archive; prior evidence and LLM-token history remain available.

## Remote LLM launcher refuses to start

Full provenance needs a local model path and manifest. For development only:

```bash
export TREX_ALLOW_UNVERIFIED_LLM=1
```

The run will be labeled non-benchmark-reproducible.
