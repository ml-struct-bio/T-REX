# Append-only archive

Each campaign archive stores one JSON object per line and one file per record
type. The principal streams are:

- `target_constraints.jsonl`
- `result_records.jsonl`
- `evidence_summaries.jsonl`
- `metric_calibrations.jsonl`
- `runtime_buckets.jsonl`
- `hypothesis_cards.jsonl`
- `action_candidates.jsonl`
- `supervisor_decisions.jsonl`
- `llm_call_records.jsonl`
- `launch_decisions.jsonl`
- `dispatch_records.jsonl`
- `route_records.jsonl`
- `panel_selections.jsonl`

The archive is append-only during a campaign. Readers tolerate a truncated final
line so a worker interruption does not invalidate preceding records. Resume
runs append to the same archive and preserve prior evidence and LLM-token usage.

`run_provenance.json` is written before dispatch. It is separate from JSONL
records because it describes the immutable run contract: source and model
hashes, target files, enabled methods, thresholds, resource layout, seed, and
environment metadata.

## Runtime-derived layout

Do not maintain a separate handwritten field list. Generate the current layout
from the installed T-ReX dataclasses:

```bash
trex-analyze schema
trex-analyze schema --json > archive_schema.json
```

The JSON payload uses schema `trex.archive-layout.v1` and contains stream
descriptions, field types, required flags, logical keys, join keys, and explicit
cross-stream relationships. The source of truth is `trex/schemas.py` plus the
record-to-file mapping in `trex/archive.py`; `trex/archive_schema.py` is the
presentation layer over those definitions.

## Identity and joins

- `result_id` is archive-unique and duplicate appends are rejected.
- `launch_id` is the preferred key from `launch_decisions.jsonl` to
  `dispatch_records.jsonl`.
- `candidate_id` joins an action to launch/dispatch records, but resumed legacy
  archives can contain recurrence; run `trex-analyze validate` first.
- `ResultRecord.parent_ids` is a polymorphic lineage array, not a result-only
  foreign key. Its first value is normally the spawning `candidate_id`; a
  concrete input `result_id`, when present, is normally second. Legacy/imported
  rows can contain external lineage tokens. Explode the array and classify each
  token against both ID domains before joining.
- `hypothesis_ids` and `selected_ids` are arrays. Explode them before joining to
  `hypothesis_id` and `result_id`, respectively. A fixed set of deterministic
  candidate-origin tags (`warmstart`, `evidence_fallback`,
  `evidence_fallback_dead_probe`, `diagnostic_i4_mcts`, `route_value_replay`,
  and `cross_family_escape`) deliberately has no HypothesisCard row;
  `trex-analyze validate` distinguishes these from broken references.
- `tick_id` aligns decision-time views but is not a molecular identity.

## Validation and evolution

`trex-analyze validate` parses every JSONL stream, reconstructs every row using
the installed frozen dataclass, checks cross-stream identity warnings, and
verifies supported provenance plus linked campaign-input hashes. Its JSON report
also classifies result-lineage tokens as candidate, result, ambiguous, or
external/unknown references. A truncated or malformed line is reported while
preceding valid rows remain readable.

Archive rows do not carry a per-row schema envelope. Therefore retain the
matching `run_provenance*.json`, T-ReX source digest or release, and the saved
`archive_schema.json` with every downstream dataset. New optional dataclass
fields remain readable by older archives; required-field or semantic changes
must be treated as a new analysis contract rather than silently rewriting rows.
