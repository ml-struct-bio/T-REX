# Running on Slurm

Run all shell commands from the repository root.


This is an alternative launch path for the supplied HPC template. It starts
vLLM on GPU 0 and worker processes on the remaining GPUs using explicit
`TREX_*` environment variables. The `.env` file is not required by the YAML
path above, and it is never loaded automatically. Do not point the YAML and
Slurm launchers at the same new archive simultaneously.

```bash
cp .env.example .env
# Edit every /path/to/... value.
set -a
source .env
set +a
mkdir -p slurm_logs
```

For a registered target, fail-closed preflight checks the target ID, configured
chains, hotspot residues, PDB SHA256, enabled backends, checkpoint manifests,
declared checkpoint files, executables, the model manifest, and optionally
pinned backend revisions:

```bash
trex-validate \
  --target cd45 \
  --asset-root "$TREX_TARGET_ASSET_ROOT" \
  --enabled-families "$TREX_ENABLED_FAMILIES" \
  --require-backends \
  --require-model \
  --verify-backend-revisions
```

For a publication release audit, add `--verify-checkpoint-content`. This rereads
every declared Qwen, AF2, and ProteinMPNN file and compares its SHA256; normal
startup uses the manifest plus file names and byte sizes to avoid repeatedly
streaming large weight trees from shared storage.

Only enabled families are required. For example, disabling `boltzgen` removes
the BoltzGen executable requirement; Foldseek remains required because it
defines the primary endpoint.

### Submit the Slurm job

The reference layout uses four H100s: GPU 0 serves the local LLM and GPUs 1–3
are workers. All public entry points default to a **48-hour cumulative
controller limit**. With three workers, the nominal reporting budget is
**144 worker GPU-hours**.

The Slurm template reserves **49 hours**, providing one hour of headroom for
startup and shutdown around the controller window. This scheduler reservation
is distinct from the 144-worker-GPU-hour reporting denominator; actual
resource usage is retained separately. If startup and shutdown exceed the
headroom, the scheduler can still interrupt the run. YAML and shell launches
require the user to arrange sufficient scheduler time.

The controller checks the cutoff at the start of each loop and then
drains busy workers sequentially, waiting up to 10 minutes per slot before
salvaging results and terminating an unfinished process. This is not a hard
per-start deadline or a shared one-hour drain. The controller uses that
behavior. See the [timing contract](reproducibility.md#execution-timing-and-shutdown).

The controller itself scales to any unique comma-separated worker GPU list. To
run more than the four-GPU reference layout, request the larger allocation from
Slurm and disable the publication-template worker-count guard, for example:

```bash
(
  export TREX_WORKER_GPUS=1,2,3,4,5,6,7
  export TREX_REQUIRE_THREE_WORKERS=0
  export TREX_CHARGED_GPUS=8
  sbatch --gres=gpu:8 --export=ALL slurm/trex_per_target_node.slurm
)
```

Export comma-separated values in the shell as shown above. Do not embed
`TREX_WORKER_GPUS` or `TREX_ENABLED_FAMILIES` inside `sbatch --export=...`:
Slurm treats their commas as separators and can silently truncate the value.

Validate the larger worker-GPU mapping first without loading the LLM or
scientific backends:

```bash
(
  export TREX_WORKER_GPUS=1,2,3,4,5,6,7
  export TREX_TEST_PYTHON=/path/to/trex-dev/bin/python
  sbatch --gres=gpu:8 --export=ALL slurm/multigpu_smoke.slurm
)
```

For a bounded backend smoke, see
[installation checks](installation.md#optional-smoke-checks). Temporary settings belong in a
subshell so they do not change the subsequent production submission.

Submit the reference campaign after the checks above:

```bash
(
  export TARGET=cd45
  export TREX_MAX_WALL_H=48.0
  sbatch --export=ALL slurm/trex_per_target_node.slurm
)
```

Site-specific account, partition, reservation, and QOS options should be passed
to `sbatch`; they are deliberately not hard-coded. The script:

1. validates the full execution contract;
2. verifies the model manifest;
3. writes `run_provenance.json`;
4. starts and health-checks vLLM on GPU 0; and
5. starts the worker controller over `TREX_WORKER_GPUS`.

For a new run, the archive is created under
`$TREX_ARCHIVE_BASE/trex_<target>_s<seed>_<YYYYMMDD_HHMM>/<slurm-job-id>/`.
Use that directory with the status and analysis commands below.

To resume the same append-only campaign:

```bash
(
  export TREX_RESUME_ARCHIVE=/absolute/path/to/existing/archive
  export TREX_MAX_WALL_H=48.0
  sbatch --export=ALL slurm/trex_per_target_node.slurm
)
```

Resume preserves the archive and writes a separate
`run_provenance_resume_<job-id>.json`. `TREX_MAX_WALL_H` is the total
controller limit: the controller recovers the elapsed offset from the
archive, so 48.0 does not add another 48 hours to a resumed campaign. Preserve
the original run's configured limit when resuming; do not replace an explicit
custom limit with the new-run default.

## Local or externally hosted LLM endpoint

`scripts/run_controller.sh` accepts an existing OpenAI-compatible endpoint.
For a benchmark-reproducible run, also provide the local model path and
manifest so provenance can identify and validate its files:

```bash
export TARGET=cd45
export TREX_TARGET_PDB=/absolute/path/to/CD45.pdb
export TREX_ARCHIVE_ROOT=/absolute/path/to/run
export TREX_LLM_BASE_URL=http://127.0.0.1:8000/v1
export TREX_QWEN_MODEL_PATH=/absolute/path/to/Qwen3.6-27B-FP8
./scripts/run_controller.sh
```

For development against a remote model whose files cannot be inspected, set
`TREX_ALLOW_UNVERIFIED_LLM=1`. The launcher prints a warning and the resulting
campaign is **not** benchmark-reproducible.
