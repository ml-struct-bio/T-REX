# Outputs and analysis

Use the actual run directory created by the [Slurm launcher](slurm.md):

```bash
TREX_RUN_ARCHIVE=/absolute/path/to/trex_cd45_s0_TIMESTAMP/JOB_ID
```

## Files to inspect

| File or directory | Contents |
| --- | --- |
| `result_records.jsonl` | Per-output measurements, execution status, ancestry and artifact paths; includes failures and incomplete results |
| `evidence_summaries.jsonl` | Full `EvidenceSummary` at each planning update, including quality, novelty, route cost/yield, pending evaluations and workers |
| `hypothesis_cards.jsonl` | `HypothesisCard` proposals and appended lifecycle updates |
| `action_candidates.jsonl` | Concrete candidates, parent links, configuration deltas and build checks |
| `supervisor_decisions.jsonl` | Ranked candidate IDs, allocation mixture and selection context |
| `llm_call_records.jsonl` | Prompt hashes, model IDs, token use, latency and validation/fallback status; not complete prompt text |
| `launch_decisions.jsonl` | Queue-admission intent and rejection reasons |
| `dispatch_records.jsonl` | Actual starts, temporary deferrals and failures, linked to candidate and launch IDs |
| `panel_selections.jsonl` | Nominated designs and panel checks, when emitted |
| Optional `metric_calibrations.jsonl`, `runtime_buckets.jsonl`, `target_constraints.jsonl`, `route_records.jsonl` | Additional supported record types; not every campaign writes every stream |
| `run_provenance*.json`, `repository_patches/` | Source, target, model, software and run settings |
| `controller_checkpoint.json` | State used for resume |
| `worker_outputs/` | Per-job logs and most backend outputs, referenced by result records |
| `$TREX_COMPLEXA_REPO/inference/` (outside the archive) | Complexa's native structures and score tables; the worker log remains in `worker_outputs/` |
| `vllm.out`, `vllm.err` | LLM server logs |

JSONL streams append records without overwriting earlier lines. Checkpoints and
cache files instead describe the latest state and can be replaced. A stream is
created on its first append; its absence is not by itself a failed run. Use the analysis
commands for joined views; `trex-analyze schema --json` describes every stream
and its fields without requiring a separate schema document.

## CD45 record examples and joins

These abbreviated objects illustrate the schema; the numbers and identifiers
are synthetic and are not experimental CD45 results. Actual files store each
object on one line and include additional fields. One completed job can append
several `ResultRecord` objects, and a failed job can retain cost with no metrics.

`result_records.jsonl` excerpt:

```json
{
  "result_id": "cd45_eval_example",
  "target_id": "05_CD45",
  "tick_id": "tick_000003",
  "backend_family": "structure_refilter",
  "parent_ids": ["candidate_eval_example", "cd45_parent_example"],
  "metrics": {"pLDDT": 92.0, "iPAE": 0.20, "binder_scRMSD": 1.2},
  "gpu_h": 0.10,
  "exit_status": "ok",
  "artifacts": {"pdb": "/outputs/example/worker_outputs/eval_example/predicted.pdb"}
}
```

`evidence_summaries.jsonl` excerpt after that result is collected:

```json
{
  "tick_id": "tick_000004",
  "target_id": "05_CD45",
  "state_label": "low_evidence",
  "strict_count": 1,
  "run_su_count": 1,
  "run_su_count_delta": 1,
  "foldseek_su_status": "ok",
  "foldseek_su_coverage": 1.0,
  "structure_dedup_scope": "binder_chain",
  "worker_gpu_h_total": 0.4,
  "worker_wall_gpu_h_total": 0.6,
  "worker_wall_gpu_count": 3
}
```

Here `strict_count` counts qualified designs and `run_su_count` counts qualified
structural clusters. The latter requires actual successful clustering; passing
the three metric cutoffs alone does not establish novelty. Missing measurements
remain missing. `worker_gpu_h_total` aggregates recorded job costs, while
`worker_wall_gpu_h_total` accounts for elapsed time across worker slots.

Follow `candidate_id` from `action_candidates.jsonl` to launch and dispatch
records. `hypothesis_ids` on the candidate link to `hypothesis_cards.jsonl`.
`ResultRecord.parent_ids` can contain both the spawning candidate ID and an
upstream result ID; match each value to the appropriate stream before joining.
The result's `tick_id` normally identifies its launch update, so a result can
first contribute to an evidence summary at a later tick. Hypothesis versions
share an ID; retain archive order when interpreting feedback.

Use `trex-analyze trace` to perform these joins, and inspect a complete synthetic
archive by running the [CPU example](../examples/analysis_demo/README.md).

## Monitor and validate

```bash
trex status "$TREX_RUN_ARCHIVE"
trex-analyze validate --archive-root "$TREX_RUN_ARCHIVE"
trex-analyze summary --archive-root "$TREX_RUN_ARCHIVE"
trex-analyze trace --archive-root "$TREX_RUN_ARCHIVE" --limit 10
```

Add `--json` to the analysis commands for machine-readable output. Validation
checks archive consistency; it does not establish a completed molecular run.
The trace joins recent evidence, proposals, selection and worker execution.
A queued or approved job is not necessarily a started job.

## Export ranked structures and measurements

The public command infers the recorded target ID:

```bash
trex export "$TREX_RUN_ARCHIVE" --n 100
```

The export contains `manifest.csv`, `manifest.json` and copied structures in
`pdbs/`. The CSV includes rank, result ID, qualification measurements, chain
identities and structure paths. It contains up to 100 qualified designs with
accessible structures; it may contain structural duplicates. It does not
produce a FASTA file. Use a new output directory for another export, or inspect
`trex export --help` for its explicit overwrite option. The standalone
`trex-export` compatibility command accepts the same detailed export settings.

Archive artifact paths can be absolute. Moving only the JSONL files does not
move or repair those references. Export before removing the original worker
outputs; share the complete export directory. In particular, copying only the
archive does not copy Complexa's external `inference/` outputs. Follow the
`artifacts` paths in each result when preserving a complete raw campaign.

## Select a diverse panel

With Foldseek and MMseqs2 configured:

```bash
set -a
source .env
set +a
trex-panel --archive-root "$TREX_RUN_ARCHIVE" --target-id 05_CD45 \
  --panel-size 8 --foldseek-binary "$TREX_FOLDSEEK_BIN" \
  --mmseqs-binary "$TREX_MMSEQS_BIN" --su-tm-score 0.60 \
  --collapse-tm-score 0.80 --sequence-identity 0.90 --sequence-coverage 0.80 \
  > "$TREX_RUN_ARCHIVE/panel.json"
```

This recomputes clustering and returns panel-selection JSON; it does not copy
structures. The helper's 0.80 whole-archive collapse setting is separate from
its 0.60 strict-SU threshold and the controller's live 0.60 collapse setting.
Neither a limited panel nor a best-N export is the full campaign SU count.

## Interpret the results

A qualified design passes all three required measurements; an SU is a
structurally distinct qualified design under binder-chain Foldseek clustering.
Missing measurements are not measured failures or zeros.
`trex-analyze summary` reads the latest recorded evidence; it does not rerun
qualification or clustering over every archived result. Its throughput uses
measured worker-wall GPU-hours.

In summary JSON, `llm_usage.calls` counts parsed archive records, which can
include skipped decisions and deterministic guard records. It is not an API
invocation count. Token totals sum recorded values; missing tokens cannot
establish zero usage.

New structure-bearing results store verified binder/target output roles in
`artifacts.output_chain_map` and `bins.output_chain_identity`. Unresolved
identities cannot receive SU credit or enter a production panel/export.
Do not infer the output binder chain from the input chain label.

Keep the complete campaign directory and its referenced files when archiving a
run. The provenance files record the actual software, input and model identities;
a seed alone does not capture asynchronous execution or guarantee identical outputs.
