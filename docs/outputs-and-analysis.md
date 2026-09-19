# Outputs and analysis

Every campaign uses an append-only archive. Per-job outputs and links are under
`worker_outputs/`; recorded artifact references identify the available files.
Controller records are separated by type into JSONL
streams so proposals, selection, dispatch, scoring, and outcomes can be audited
without parsing console logs.

Start with the [result interpretation guide](result-interpretation.md) for the
difference between recorded counts, missing observations, and reproduction
claims. Prefer versioned JSON for scripts; human-readable labels may be clarified
without changing the JSON schema or archived data.

## Read-only campaign summary

```bash
trex-analyze summary --archive-root /path/to/archive
trex-analyze summary --archive-root /path/to/archive --json
```

JSON summaries use schema `trex.analysis-summary.v1`. The legacy
`llm_usage.calls` and `llm_usage.by_role.<role>.calls` fields count parsed archive
records, **not API invocations**. They can include deterministic advisory guard
records and skipped decisions. Human-readable output labels this value
`LLM records: count=...`; JSON names and values remain unchanged.

Token totals are sums of recorded values, not independently verified provider
billing. Missing token values contribute zero to this legacy summary, so a zero
total is not proof that a model request never occurred. Inspect each record's
`role`, `model`, and `parse_status` when interpreting usage.

The live summary rate uses `worker_wall_gpu_h_total`, not reserved allocation hours.
`worker_gpu_h_total` is completed-job cost used for route attribution.
Raw reserved-allocation metadata may be present in archives for audit, but it
is not reported as a SU/GPU-hour rate.

The summary reads the latest recorded evidence; it does not recompute a final
publication endpoint from all generated artifacts. The fixed paper denominator
and the live measured denominator have separate meanings. A newly created
archive may not yet contain result or evidence streams. Missing measurements
are not zeros, and a valid empty archive is not evidence of a completed run.

The summary also reports:

- strict and structure-unique counts;
- Foldseek status, coverage, and binder-chain scope;
- sequence clustering status;
- results and exits by backend family;
- launch and dispatch realization;
- started actions by family and exploit/rescue/explore mode; and
- LLM input/output tokens separately, including role-level totals.

`dispatch_status` is the literal immutable archive vocabulary. The additional
`dispatch_outcome` count is the operator-facing interpretation: it separates
`capacity_deferred` and `cancelled_before_start` from genuine
`dispatch_failed` rows while retaining the raw count for compatibility. The
same derived `outcome` appears beside raw `status` in JSON decision traces.

## Archive integrity

```bash
trex-analyze validate --archive-root /path/to/archive
trex-analyze validate --archive-root /path/to/archive --json
```

The versioned validation payload (`trex.archive-validation.v1`) checks malformed
JSON, required dataclass fields, duplicate result IDs, mixed target IDs,
unsupported provenance versions, prompt hashes, and campaign input/resolved
artifact hashes. It rejects panels that reference absent results and classifies
polymorphic `ResultRecord.parent_ids` tokens for downstream lineage analysis.
Warnings identify absent optional streams, unmatched legacy joins, incomplete
Foldseek coverage, or an unavailable sequence-diversity pass.

Deterministic candidate origins such as `warmstart`, `evidence_fallback`, and
`route_value_replay` are not HypothesisCard foreign keys. Validation recognizes
the fixed system-origin vocabulary and still warns for every other unresolved
`hypothesis_ids` value.

## Evidence-linked decision trace

```bash
trex-analyze trace --archive-root /path/to/archive --limit 10
trex-analyze trace --archive-root /path/to/archive --limit 10 --json
```

The default is a concise human view. `--json` returns versioned
`trex.decision-trace.v1` envelope with `trex.decision-trace-entry.v1` entries,
including an explicit empty `entries` array when no trace is available.

For each recent tick the output joins:

- state class and diagnostic driver;
- concise `HypothesisCard` claim and structured reasoning trace;
- Supervisor mode mixture and deterministic clamps;
- validated action family/configuration, launch decision, and explicit candidate
  join status (`unique`, `missing`, or `ambiguous`); and
- actual `DispatchRecord`.

This is the supported audit view for “evidence -> hypothesis -> proposed action
-> validated dispatch.” It does not expose or reconstruct hidden
chain-of-thought.

## Machine-readable archive contract

```bash
trex-analyze schema
trex-analyze schema --json
```

This contract is generated from the runtime dataclasses. It lists every stream,
field type, required field, logical key, join key, and cross-stream relationship.
List-valued keys must be exploded before a tabular join. In particular,
`parent_ids` is polymorphic: it normally contains a spawning `candidate_id` and
can also contain an input `result_id`; it must not be treated as a result-only
foreign key.

## Other machine-readable command contracts

All structured command outputs have a named schema so notebook and workflow
consumers can reject incompatible changes instead of guessing their shape.

| Command | Output schema |
| --- | --- |
| `trex campaign preflight CONFIG --json` | `trex.campaign-preflight.v1` |
| `trex-validate ... --json` | `trex.installation-validation.v1` |
| `trex-target list --json` | `trex.target-list.v1` |
| `trex-target resolve ... --format json` | `trex.resolved-target.v1` |
| `trex-backend list --json` | `trex.backend-list.v1` |
| `trex-provenance capture ...` | `trex.provenance-capture.v1` |

## Final panel

```bash
trex-panel \
  --archive-root /path/to/archive \
  --target-id 05_CD45 \
  --panel-size 8 \
  --foldseek-binary "$TREX_FOLDSEEK_BIN" \
  --mmseqs-binary "$TREX_MMSEQS_BIN" \
  --su-tm-score 0.60 \
  --collapse-tm-score 0.80 \
  --sequence-identity 0.90 \
  --sequence-coverage 0.80
```

The 0.80 whole-archive collapse setting preserves the original post-hoc helper;
it is separate from the live controller's default and from strict-SU counting.
Changing it is an explicit analysis choice, not a behavior-neutral refactor.

The command reruns strict-only Foldseek and MMseqs2 in memory because historical
`ResultRecord` rows are immutable. It refuses to select a strict panel when
trusted structural deduplication is incomplete. Add `--append` only when the
final selection should become another immutable archive record.

Stdout uses schema `trex.final-panel.v1` and records every clustering threshold.
With `--append`, an existing `panel_id` is rejected instead of adding an
ambiguous duplicate selection.

## Export structures

```bash
trex-export \
  --archive-root /path/to/archive \
  --target-id 05_CD45 \
  --n 100 \
  --su-tm-score 0.60 \
  --sequence-identity 0.90 \
  --sequence-coverage 0.80 \
  --out-dir /path/to/export
```

Export is a copy operation; it does not change archived metrics, clusters, or
candidate identities.

`manifest.json` uses schema `trex.best-n-manifest.v1` and contains the archive,
target, ranking rule, gates, clustering thresholds, diagnostics, and designs.
`manifest.csv` is the flat spreadsheet view; copied structures are under `pdbs/`.

Command stdout uses `trex.best-n-export.v1` and reports the resolved manifest
paths. The CSV is always created with a stable header, even for zero designs.
To prevent stale PDBs or manifests from mixing across analyses, a nonempty
output directory is rejected. `--overwrite` replaces only recognized T-ReX
export files and still refuses a directory containing unrelated files. The new
export is staged completely before publication, so an I/O failure preserves the
previous complete export.

## Downstream analysis

Validate an archive before analysis, and save the generated layout contract with
the analysis outputs:

```bash
trex-analyze validate --archive-root /path/to/archive
trex-analyze schema --json > archive_schema.json
```

For ready-to-save JSON views, use the
[analysis handoff below](#save-an-analysis-handoff).
Keep `validation.json`, `summary.json`, the schema and source provenance with
analysis outputs. A limited decision trace is a recent-history view.

JSONL streams load directly into dataframe tools. The following optional pandas
example assumes a populated archive with the named streams; pandas is an
optional notebook dependency, not a controller requirement. It flattens nested
measurements and performs a checked join:

```python
from pathlib import Path

import pandas as pd

root = Path("/path/to/archive")
results = pd.read_json(root / "result_records.jsonl", lines=True)
metric_columns = pd.json_normalize(results.pop("metrics")).add_prefix("metric.")
results = results.join(metric_columns)

launches = pd.read_json(root / "launch_decisions.jsonl", lines=True)
dispatches = pd.read_json(root / "dispatch_records.jsonl", lines=True)
realized = (
    dispatches.merge(
        launches[["launch_id", "candidate_id", "status", "why"]],
        on="launch_id",
        how="left",
        validate="many_to_one",
    )
)

lineage = (
    results[["result_id", "parent_ids"]]
    .explode("parent_ids")
    .dropna(subset=["parent_ids"])
    .rename(columns={"parent_ids": "lineage_token"})
)

candidates = pd.read_json(root / "action_candidates.jsonl", lines=True)
candidate_ids = set(candidates["candidate_id"])
result_ids = set(results["result_id"])


def reference_kind(token):
    in_candidates = token in candidate_ids
    in_results = token in result_ids
    if in_candidates and in_results:
        return "ambiguous"
    if in_candidates:
        return "candidate"
    if in_results:
        return "result"
    return "external_or_unknown"


lineage["reference_kind"] = lineage["lineage_token"].map(reference_kind)
```

Recommended relationships are:

- `tick_id` for evidence, Supervisor, LLM-call, launch, dispatch, and result time;
- `launch_id` for selected intent to actual dispatch realization;
- classified, exploded `parent_ids` for candidate-to-result and result-to-result
  lineage; and
- `result_id` with exploded `PanelSelection.selected_ids` for panel membership.

Use `candidate_id` for proposal/config joins only after validation reports no
recurrence; `launch_id` is the preferred launch-to-dispatch key.

## Safe handling

- Never edit JSONL rows in place.
- Resume into the same root only through the controller.
- Put post-hoc analyses outside the archive or in a clearly named analysis
  subdirectory.
- Keep provenance files and repository patches with any released archive.
- Report Foldseek threshold, structure scope, worker-GPU-hour denominator, and
  missing-score coverage with every endpoint.

## Save an analysis handoff

Save machine-readable views outside the campaign archive:

```bash
(
  set -e
  ARCHIVE_ROOT=/absolute/path/to/archive
  ANALYSIS_DIR=./analysis/example_run
  mkdir -p "$ANALYSIS_DIR"
  trex-analyze validate --archive-root "$ARCHIVE_ROOT" --json > "$ANALYSIS_DIR/validation.json"
  trex-analyze summary --archive-root "$ARCHIVE_ROOT" --json > "$ANALYSIS_DIR/summary.json"
  trex-analyze trace --archive-root "$ARCHIVE_ROOT" --limit 10 --json > "$ANALYSIS_DIR/recent_trace.json"
  trex-analyze schema --json > "$ANALYSIS_DIR/archive_schema.json"
)
```

Validation failure stops the block; read `validation.json` before continuing.
`summary.json` describes recorded evidence at the latest available snapshot;
it is not an automatic recomputation of the paper's final endpoint.
`recent_trace.json` contains the requested recent decisions, not the entire
history. Keep provenance and source records with downstream results.

For tabular analysis, start from `result_records.jsonl`. Use `result_id` as
the result key and the schema's documented relationships for joins.
`parent_ids` can reference actions as well as results. Missing measurements
remain missing rather than becoming zero. See the
[downstream notebook example](#downstream-analysis).
