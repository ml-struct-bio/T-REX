# Campaign input and output interface

The `trex campaign` interface gives each run one reviewable input and a compact
view of its outputs. It is a user-interface layer over the existing scientific
controller; qualification, structural clustering, candidate selection,
evaluation admission, dispatch, and append-only archive semantics are
unchanged.

## Workflow

```bash
trex campaign init campaign.yaml \
  --target cd45 \
  --asset-root /absolute/path/to/target-assets \
  --archive-root ./runs/cd45

trex campaign show campaign.yaml
trex campaign preflight campaign.yaml --strict --require-model
trex campaign run campaign.yaml --dry-run
trex campaign run campaign.yaml
trex campaign status ./runs/cd45
```

`init` writes every ordinary policy field instead of hiding important defaults.
Edit the generated file, then use `show` to see absolute target and output
paths, effective action families, worker GPUs, LLM identity, scientific policy,
and any resolution notes. `show --json` produces the same information as a
machine-readable object. `show --controller-command` shows the equivalent
legacy controller invocation.

`preflight --json` uses the versioned `trex.campaign-preflight.v1` envelope and
the nested `trex.preflight-report.v1` check schema.

`preflight` never starts an LLM or scientific worker. Without `--strict`,
missing external backends are warnings so a configuration can be inspected on
a laptop or login node. `--strict` makes every enabled backend, executable, and
required checkpoint fatal. `--require-model` additionally validates the local
Qwen model path and manifest. A nonzero exit status means at least one required
check failed.

`run --dry-run` is read-only: it resolves and preflights the campaign but does
not create the archive. A real `run` requires enabled scientific backends and a
local LLM model with a valid manifest, records input plus provenance artifacts,
and only then enters the asynchronous controller. No worker starts if provenance
capture fails. `--require-model` remains useful on a dry run; it is automatic on
a real run.

Default model checks validate the manifest, file names and byte sizes.
`--verify-checkpoint-content` additionally rehashes the declared files.
The endpoint check validates URL syntax; it does not test server reachability.

## YAML schema

The schema version is `trex.campaign.v1`. Unknown keys are errors. Relative
paths are relative to the YAML file, not the caller's current directory.

```yaml
schema_version: trex.campaign.v1
name: cd45-replicate-01

target:
  name: cd45
  asset_root: /data/trex-target-assets
  constraint: null       # optional registered-target override
  pdb: null              # optional registered-target override

run:
  archive_root: ./runs/cd45-replicate-01
  max_wall_hours: 48.0
  seed: 1
  worker_gpus: ['1', '2', '3']
  enabled_families:
    - complexa_beam
    - bindcraft
    - structure_refilter

llm:
  base_url: http://127.0.0.1:12000/v1
  model: vllm/Qwen/Qwen3.6-27B-FP8

memory:
  cross_campaign_path: null  # optional cross-campaign-memory-v3 JSON

policy:
  critic: true
  evidence_skip: false
  exemplars: true
  foldseek_su_tm_score: 0.60
  foldseek_collapse_tm_score: 0.60
  selector_quota_realization: fractional_carry
  selector_mode_window_k: 1
  selector_adaptive_mode_window_k: false

backends:
  repo_root: /path/to/T-ReX
  external_root: /path/to/external
  complexa_repo: /path/to/Proteina-Complexa
  legacy_complexa_repo: /path/to/Proteina-Complexa-community
  complexa_python: /path/to/Proteina-Complexa/.venv/bin/python
  bindcraft_repo: /path/to/BindCraft
  bindcraft_env: /path/to/BindCraft/environment
  boltzgen_repo: /path/to/BoltzGen
  boltzgen_binary: /path/to/BoltzGen/.venv/bin/boltzgen
  boltzgen_cache: /path/to/BoltzGen/checkpoints
  foldseek_binary: /path/to/foldseek
  mmseqs_binary: /path/to/mmseqs
  qwen_model_path: /path/to/Qwen3.6-27B-FP8
  qwen_model_manifest: /path/to/qwen-manifest.json
```

For a custom target, both `target.constraint` and `target.pdb` are required.
For a registered target, `asset_root` normally supplies its PDB while the
checked-in registry supplies the constraint. Diagnostic-only generators are
automatically paired with `structure_refilter`; this appears explicitly in
`show` and `campaign_resolved.json`.

`memory.cross_campaign_path` is the only campaign-file switch for historical
LLM memory. When configured, preflight validates the schema and records its
content SHA256. The resolved campaign, run provenance, and per-call prompt hash
therefore identify the exact advisory input. It never overrides current-target
evidence or deterministic safeguards.

For compatibility, a null backend path first inherits its corresponding
`TREX_*` environment variable and otherwise expands to the documented checkout
default. Executables found through `PATH` are recorded as absolute paths, and
virtual-environment executable symlinks are deliberately preserved. Any other
ambient `TREX_*` tuning variables are listed under
`controller_interface.inherited_trex_environment` so they are not invisible.
Credential-like names containing `TOKEN`, `KEY`, `SECRET`, `PASSWORD`, or
`CREDENTIAL` are recorded as `<redacted>`.

## Execution timing

New campaign files, omitted fields and the controller CLI use a cumulative
48-hour controller limit. Explicit values in existing YAML files are preserved.
The YAML selector-window defaults remain 1 and non-adaptive; the low-level CLI
retains its historical selector-window defaults. `show --controller-command`
displays the values actually passed. Time limits must be finite and positive.

`run.max_wall_hours` is a cumulative controller limit, not a Slurm allocation,
a guaranteed completion deadline, or the paper's GPU-hour denominator. YAML
launches do not allocate GPUs or schedule a drain window automatically. The
original loop checks the limit before each iteration, then uses a sequential
10-minute-per-busy-slot shutdown grace. Resume includes archived elapsed time.
The Slurm template reserves 49 hours for the 48-hour controller window plus
startup/shutdown headroom. The nominal three-worker reporting budget is 144
worker GPU-hours; scheduler reservation and measured exposure are distinct.
See the [timing contract](reproducibility.md#execution-timing-and-shutdown).

## Output contract

| Output | Purpose | Authority |
| --- | --- | --- |
| `campaign_input.yaml` | Exact user-supplied bytes | Human input audit |
| `campaign_resolved.json` | Absolute paths, effective families, model, policy, controller arguments, compatibility environment | Effective execution input |
| `run_provenance*.json` | Source, model, checkpoint, target, runtime, and repository hashes | Reproducibility audit |
| `result_records.jsonl` | Results, measurements, identities and artifact references | Recorded outputs |
| `hypothesis_cards.jsonl`, `action_candidates.jsonl` | Hypotheses and proposed actions | Proposal history |
| `supervisor_decisions.jsonl`, `launch_decisions.jsonl`, `dispatch_records.jsonl` | Priorities, scheduling decisions and worker realization | Decision/execution history |
| `llm_call_records.jsonl` | Recorded model/guard events and available token usage | Invocation audit; inspect status and role |
| `evidence_summaries.jsonl` | Immutable per-update derived campaign view | Decision-time evidence |
| `controller_checkpoint.json` | Resume and live progress state | Runtime durability |
| `trex campaign status` | Compact counts, current-schema evidence metrics, execution state, typed-read warnings | Convenience view only; use `trex-analyze validate` for full integrity |

Status output reports `input_artifact_status=campaign_resolved` for YAML-driven
runs and `provenance_only` for the supported Slurm/environment launcher. In the
latter case it recovers target, budget, worker GPUs, and enabled families from
`run_provenance*.json`; the absence of `campaign_resolved.json` is therefore
not mislabeled as an integrity failure. Invalid or entirely missing input
artifacts remain warnings.

The `execution` section preserves literal `dispatch_status` counts and adds
`dispatch_outcome` counts for operator-facing deferred/cancelled/failure
semantics.

New provenance records use `v7.3.3_run_provenance_v2`. Archive validation
retains explicit v1 compatibility, rejects unknown schema versions, and
rehashes the linked campaign input artifacts. `trex-analyze schema --json`
provides the installed archive fields and join relationships.

If a resume changes either the YAML or its effective resolved input, the
original pair is preserved and the new input and resolved files receive the
same SHA256-derived suffix. The convenience status view never changes or
replaces archive records.

Every command supporting `--json` writes only the structured object to stdout,
which allows validation and monitoring scripts to consume it without parsing
the human table.

At execution time the resolved `backends` section becomes one immutable
`RuntimePaths` object passed directly to the controller. The same resolved
paths remain in `campaign_resolved.json`; `TREX_*` variables are retained for
legacy adapters and external backend extensions.

## Code ownership

The interface is split by responsibility:

- `trex/campaign/models.py`: typed input and resolved execution contracts
- `trex/campaign/config.py`: YAML parsing, validation, and target resolution
- `trex/campaign/preflight.py`: read-only execution checks
- `trex/campaign/artifacts.py`: immutable input artifact writing
- `trex/campaign/run_lock.py`: single-launcher archive lock
- `trex/campaign/provenance.py`: fail-closed run provenance and resume naming
- `trex/campaign/status.py`: read-only archive projection
- `trex/campaign/presentation.py`: terminal rendering
- `trex/campaign/cli.py`: command handlers only
- `trex/campaign/controller_args.py`: legacy CLI compatibility boundary
- `trex/campaign/runtime/worker.py`: worker-slot lifecycle state
- `trex/campaign/runtime/timeouts.py`: timeout policy and diagnostics
- `trex/campaign/runtime/scheduling.py`: controller polling cadence
- `trex/campaign/runtime/paths.py`: immutable backend path resolution
- `trex/prompt_catalog.py`: runtime-backed prompt inspection and audit metadata
- `trex/resource_paths.py`: checkout-or-wheel publication metadata resolution
- `trex/archive_schema.py`: runtime-derived stream fields and join relationships
- `trex/analysis.py`: read-only summary, integrity, trace, and schema CLI
- `trex/dispatch_outcomes.py`: raw-to-operator dispatch outcome interpretation
- `trex/finalize_panel.py`: reproducible diversity-panel materialization
- `trex/export_best_n.py`: versioned ranked manifest and structure export
- `trex/backends/bindcraft.py`: BindCraft settings and direct argv builder
- `trex/backends/af2.py`: canonical AF2-refilter argv, seed, and environment
- `trex/backends/boltzgen.py`: BoltzGen YAML, hotspot mapping, and launch builder
- `trex/backends/complexa.py`: shared Complexa override and command builder
- `trex/backends/proteinmpnn.py`: ProteinMPNN redesign launch builder
- `trex/backends/process_management.py`: process-group and environment isolation
- `trex/execution/result_processing.py`: parser routing and result normalization
- `trex/execution/dispatch.py`: pending queue admission, retry, worker-slot assignment, and dispatch audit
- `trex/execution/worker_supervision.py`: completion, timeout salvage, and shutdown drain lifecycle
- `trex/execution/score_conversion_scheduling.py`: deterministic canonical-score follow-up planning
- `trex/execution/circuit_breaker.py`: evidence-driven family disabling and pending-queue deltas
- `trex/execution/progress_checkpoint.py`: atomic resume/progress state and between-tick monitoring refresh
- `trex/tick/config.py`: immutable configuration shared by live-tick phases
- `trex/tick/archive_snapshot.py`: append-order-preserving tick input snapshot
- `trex/tick/evidence_clustering.py`: Foldseek/MMseqs bins and deduplication provenance
- `trex/tick/evidence.py`: SU accounting, dry-timer signals, panel snapshot, and evidence reduction
- `trex/tick/proposal.py`: Planner reuse/call, critic validation, and ordered audit output
- `trex/tick/candidates.py`: parent/refilter context, candidate construction, and feasibility populations
- `trex/tick/supervision.py`: Supervisor execution context, call/reuse, and fallback semantics
- `trex/tick/selection.py`: mode hints, route saturation, and deterministic Selector input
- `trex/tick/lifecycle.py`: pre-Planner TTL retirement and descendant-to-baseline lifecycle updates
- `trex/tick/summary.py`: stable evidence-only and full-tick output projections
- `trex/selection/policy.py`: Supervisor/fallback mixture, clamps, priority, and mode resolution
- `trex/selection/admission.py`: feasibility, capacity, family-cost, and route admission
- `trex/selection/quota.py`: deterministic quota realization, credit, repairs, and audit output
- `trex/selection/ranking.py`: diversity-aware ranking, deferred backfill, and escape floor
- `trex/selection/emission.py`: launch/rejection records and non-launch diagnostics

This separation keeps presentation, filesystem I/O, validation, and scientific
execution from accumulating in another monolithic controller module.
