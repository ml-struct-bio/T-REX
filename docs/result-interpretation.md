# Reading existing results correctly

This guide is for inspecting an existing archive. It does not launch models or
workers, regenerate results, or change the scientific configuration. For the
checkout's unresolved equivalence status, see
[Reproducibility contract](reproducibility.md#compatibility-status).

## Start with the records

Use the same archive root for both commands:

```bash
trex-analyze validate --archive-root /path/to/archive --json
trex-analyze summary --archive-root /path/to/archive --json
```

Read validation errors and warnings before interpreting the summary. The
summary is a compact view, not an integrity check: it can skip malformed JSONL
rows that validation reports. It does not rerun the underlying calculations.
Keep the archive's initial and resume provenance with any exported summary.

## What a passing check establishes

| Check | Establishes | Does not establish |
| --- | --- | --- |
| Archive validation | The implemented format, identity, and reference checks pass | Scientific correctness or agreement with the original controller |
| Frozen-table verifier | Checked-in table hashes and declared calculations agree | New experiments reproduced those values |
| Model manifest verification | The declared file checks pass; full-content mode also checks file bytes | The server loaded those bytes or generated equivalent responses |
| Prompt catalog | Available implementations and their recorded identities | Which roles were actually invoked in a historical run |
| Limited source parity tests | Agreement for the specified modules and test cases | Coverage of every asynchronous execution path |

Do not convert one of these claims into another. A smoke run that produces an
archive is also not, by itself, a manuscript reproduction.

## LLM names, calls, and token totals

The model fields describe different things:

- `model.client_model`: the provider-qualified identifier configured for the client.
- `model.served_name`: the declared model name served by the endpoint.
- `model.content_sha256`: the checkpoint content identity recorded by provenance.
- `LLMCallRecord.model`: the model label recorded for that decision or advisory event.

Names alone do not identify checkpoint bytes, and an endpoint accepting a model
name does not prove scientific or sampling equivalence. Missing historical
fields remain unknown; do not fill them from the current checkout's defaults.

In `trex.analysis-summary.v1`, `llm_usage.calls` and each role's `calls` are
legacy names for **record counts**, not API invocation counts. The text summary
therefore prints `LLM records: count=...`. Its JSON field names and calculations
are unchanged for existing consumers.

For example, an archive with one Planner response, one Supervisor skip, and one
deterministic guard entry contains three records, not three demonstrated model
invocations. Inspect `role`, `model`, and `parse_status` together. In particular,
`role=critic` with `model=deterministic_guard` is not an LLM Critic call. Conversely,
zero recorded tokens alone do not prove that no request was attempted: usage
can be missing after an error. Summary token totals sum recorded values and
treat missing values as zero; they are not provider billing statements.

The historical manuscript campaigns did not use the standalone LLM Critic or
cross-campaign memory. Their availability in the code must not be confused with
their historical use. Prompt metadata v2 makes the deterministic guard a
separate field; legacy metadata and records are not rewritten. See
[Prompt reproducibility](reproducibility.md#prompt-reproducibility).

## Missing observations and accounting

A missing endpoint or resource field (`null` in JSON, `None` in the text view)
is not a measured zero. Check `latest_tick` and the presence of evidence records
before treating a summary as a final endpoint. A summary from an interrupted
run is a snapshot of recorded evidence, not a completed-benchmark certificate.

The live summary uses measured worker-wall GPU-hours. The frozen paper tables
use their declared common nominal budget instead. Completed-job cost and raw
reserved-allocation hours have separate meanings; do not substitute them into
a reported rate. Preserve the numerator, denominator definition, cutoff, and
coverage metadata together. See [Resource accounting](reproducibility.md#resource-accounting).

## Preserve the audit trail

Use the versioned JSON output for notebooks and automation; human-facing labels
may be clarified without changing the JSON contract. Do not edit archive rows
to match a newer schema, replace missing values with guessed defaults, or merge
different runs without retaining their identities and provenance. A declared
configuration is evidence of configuration, while records are evidence of
observed events; neither alone proves full original-code equivalence.

For detailed field and join semantics, see
[Outputs and analysis](outputs-and-analysis.md) and
[Append-only archive](archive-schema.md).

## AF2 chain identity

An AF2 report's top-level `binder_chain` and `target_chain` describe the input
complex. The prediction can use different labels. New reports include
`prediction_chains` with output labels and per-chain sequence hashes; parsed
ResultRecords attach those output labels to `artifacts.pdb_path`. A compact
verified map travels with the record, so the original input need not remain
available for extracting that prediction's binder.

For legacy AF2 records, clustering resolves identity from the adjacent report
and the input/predicted structures, without rewriting records. Missing,
ambiguous, or mismatched identity is excluded from structural and sequence
clustering. Qualification measurements remain available; exclusion means
unresolved diversity evidence, not a failed qualification measurement. A
historical stored cluster assignment is not a fresh verification of identity.
