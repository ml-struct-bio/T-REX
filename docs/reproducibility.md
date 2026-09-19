# Reproducibility contract

A T-ReX production campaign is defined by the joint contract below. A seed or
source commit alone is insufficient.

## Scope

Campaign LLM roles are Planner and Supervisor. The campaign advisory guard is
deterministic; the standalone optional LLM Critic and cross-campaign memory were
not used in the primary manuscript campaigns.

AF2 output chain identities are verified from the predicted structure. Input
chain labels must not be assumed to identify the output binder chain. Reanalysis
of older records with unverified chain labels can change diversity evidence;
archived online decisions and frozen paper tables remain immutable.

The [result interpretation guide](result-interpretation.md) distinguishes archive
integrity, frozen-table audits, model identity and behavioral checks.

## Contract fields

1. **Controller source**: source-tree digest and release commit.
2. **Model**: served name, upstream revision, complete content manifest, and
   runtime configuration.
3. **Target**: exact PDB and constraint hashes, chain IDs, hotspots, and target
   ID.
4. **Search space**: enabled families, registered operators, allowed parameter
   ranges, and deterministic feasibility checks.
5. **Evaluation**: strict thresholds, canonical score role, Foldseek threshold,
   binder-chain scope, and sequence-clustering settings.
6. **Resources**: worker slots, worker-wall cutoff, charged allocation,
   hardware, and scheduler metadata.
7. **External software**: backend commits, tracked patches, executable paths,
   and resolved environment versions.
8. **Randomness**: campaign seed and derived per-launch seeds.

## Per-run provenance

Before any worker starts, the production wrapper writes
`run_provenance.json`. Resumes write
`run_provenance_resume_<job-id>.json` rather than replacing the first record.

The payload contains:

- T-ReX source-tree SHA256 and Git state;
- model content digest and manifest;
- target PDB/config SHA256;
- families, thresholds, seed, wall limit, worker list, and accounting fields;
- Python, OS, package, CUDA visibility, Slurm job ID, and host metadata; and
- external Git commits, dirty status, tracked-diff SHA256, and archived patches.

The source-tree digest covers `trex/`, target/reproducibility configuration,
launch scripts, the Slurm wrapper, `pyproject.toml`, the README, and the
source-backed publication prompt catalog. Git status includes untracked paths;
therefore an untracked source module cannot be reported as a clean checkout.


## Prompt reproducibility

`docs/all_prompts_snapshot.txt` catalogs the available Planner, Supervisor, and
standalone optional LLM Critic implementations. Catalog membership is not a
record of model invocation. Campaign LLM roles are Planner and Supervisor;
the campaign advisory guard is deterministic and is not the LLM Critic.
The snapshot is generated directly from the available system prompts and
dynamic user-prompt builders:

```bash
python docs/all_prompts.py > docs/all_prompts_snapshot.txt
python scripts/check_architecture.py
```

The release gate compares the checked-in snapshot byte-for-byte with the runtime
catalog. Each new `run_provenance*.json` records the available prompt identities,
source hashes, usage scopes, and declared per-role configuration. The nested
`prompts.schema_version` is `trex.prompt-catalog.v2`: the standalone Critic's
call configuration is disabled for campaigns, while
`prompts.deterministic_guard.enabled` separately records the legacy
`controller.critic_enabled` setting. An unspecified guard setting is null,
not an inferred false. These fields describe configuration, not the number of
actual calls; use role/model/status fields in `LLMCallRecord` for call accounting.

Historical provenance is not rewritten. In v1 metadata, the nested Critic
`enabled` value could reflect the deterministic guard switch; it must not be
read as evidence of an LLM Critic invocation. If `memory.cross_campaign_path`
is configured, provenance additionally stores its schema and content SHA256.
Exact per-tick dynamic prompt hashes remain in `LLMCallRecord`.

## Append-only audit trail

The archive separates:

- measured results and derived evidence;
- LLM calls and token use;
- hypothesis and action proposals;
- Supervisor decisions and deterministic clamps;
- selection intent (`LaunchDecision`);
- actual worker realization (`DispatchRecord`); and
- final panel selections.

No single console log is required to reconstruct the control flow.

## Execution timing and shutdown

All public entry points default to a cumulative 48-hour controller limit:
the Slurm wrapper, shell launcher, generated YAML, omitted YAML fields and
low-level CLI. Three worker slots correspond to the nominal 144-worker-GPU-hour
reporting budget. Explicit values in existing campaign files are preserved.
YAML and CLI time limits must be finite and greater than zero.

The Slurm template reserves 49 hours, including one hour of headroom for startup
and shutdown. The extra reservation is not added to the nominal reporting
denominator. Measured worker exposure and actual scheduler resources remain
separate audit quantities. Shell/YAML interfaces do not allocate hardware.

The controller checks elapsed time at the beginning of each loop.
Planning can take time before a later dispatch, so this is not a hard per-start
deadline. At shutdown, each busy slot is visited in order and receives a wait
of up to 600 seconds. A timeout triggers salvage parsing and process termination;
normal completion is parsed and the slot is freed. A parse error is recorded
without abandoning later slots. There is no shared one-hour completion deadline
and no extra planning or final endpoint recomputation after this drain.
Startup precedes the controller clock and consumes scheduler allocation time.
The scheduler can end the job before every sequential wait finishes.

The loop-start cutoff and sequential shutdown behavior are preserved.
Resume adds the archived elapsed offset before checking the same configured
launch-window limit; it does not grant another full window. Final paper
eligibility and its fixed denominator belong to the separate endpoint analysis,
not the last live EvidenceSummary. See [result interpretation](result-interpretation.md).

## Resource accounting

The live run-level operational rate is:

```text
run_su_count / worker_wall_gpu_h_total
```

where:

```text
worker_wall_gpu_h_total = elapsed wall hours x worker GPU slots
```

The reference campaign has three worker slots and a separate LLM GPU. In a
scaled run, `worker_wall_gpu_h_total` uses the length of `TREX_WORKER_GPUS`; for
example, seven workers for two wall hours is 14 worker-wall GPU-hours. For the
frozen seven-target paper benchmark, all endpoint counts are instead divided by
the common nominal budget of 144 H100 worker-GPU-hours per method--target
campaign. Measured worker exposure remains audit metadata and is not substituted
for that fixed benchmark denominator.

- `worker_gpu_h_total`: sum of completed worker costs, used for route-level
  learning and cost attribution.
- `worker_wall_gpu_h_total`: measured live-campaign exposure, used for the
  operational monitoring rate above.
- reserved-allocation fields and their historical `run_su_per_charged_gpu_h_*`
  rates: audit-only information about controller/LLM and idle time, retained
  for archive and LLM-input compatibility; never the decision objective or
  the primary reported throughput.

These fields and the paper benchmark's fixed denominator must not be
interchanged.

## Model and checkpoint manifests

The checked-in Qwen, AF2, and ProteinMPNN manifests include a SHA256 for every
declared file and a combined digest. Normal startup validates manifest
integrity, file names, and byte sizes:

```bash
# Qwen model
trex-provenance model-digest \
  --model-path "$TREX_QWEN_MODEL_PATH" \
  --manifest "$TREX_MODEL_MANIFEST"
```

A publication release audit rehashes every declared byte:

```bash
# Qwen model
trex-provenance model-digest \
  --model-path "$TREX_QWEN_MODEL_PATH" \
  --manifest "$TREX_MODEL_MANIFEST" \
  --full-content

# Qwen plus enabled AF2/ProteinMPNN assets
trex-validate \
  --target cd45 \
  --asset-root "$TREX_TARGET_ASSET_ROOT" \
  --enabled-families "$TREX_ENABLED_FAMILIES" \
  --require-backends --require-model \
  --verify-backend-revisions --verify-checkpoint-content
```

The second command also verifies the AF2 and ProteinMPNN trees when their
families are enabled. Create a manifest for a new snapshot with
`trex-provenance hash-model --model-path PATH --out MANIFEST.json`. Full hashing
is reserved for release audits to avoid repeatedly streaming large licensed
weight trees from shared storage.

## Resume

Set `TREX_RESUME_ARCHIVE` to the exact prior root. T-ReX appends new records and
retains earlier EvidenceSummary and token history. A resume provenance record
documents source/model/backend or resource changes. Do not concatenate archives
manually or copy only the latest EvidenceSummary.

## Deterministic versus stochastic reproduction

The unit/replay suite and frozen source-table calculations are deterministic.
Regression checks cover deterministic policy, prompt, archive and execution
contracts. Passing them does not certify every asynchronous controller or
external backend execution path.

End-to-end molecular campaigns are stochastic because they combine:

- stochastic generators;
- nondeterministic GPU kernels;
- asynchronous completion order;
- LLM sampling/runtime behavior; and
- scheduler/hardware effects.

Consequently, protocol reproduction means the same declared controller,
targets, action space, evaluation, budget, and dependencies. It does not promise
the same binder identities or an identical endpoint count.

## Published numerical results

The frozen source tables and their checksums are public under
`benchmarks/paper_7target_144h/`. Full raw-structure reproduction additionally
requires the archive deposit specified in `docs/benchmark-reproduction.md`.
