# Run a campaign

Complete [installation](installation.md) and [asset setup](assets.md), then run
commands from the T-REX checkout. The public interface keeps campaign choices in
one YAML and installation paths in `.env`.

## Create the campaign YAML

Create the installation profile once and verify its backend executable paths:

```bash
cp .env.example .env
```

Create a campaign by specifying the target, total GPUs, controller duration and
output directory:

```bash
trex init campaign_cd45.yaml \
  --target cd45 \
  --gpus 4 \
  --hours 48 \
  --output /absolute/path/to/outputs
```

The first argument is the output path for the generated YAML and may use any
name. Use a target-specific name such as `campaign_cd45.yaml`, then pass the
same path to the later commands. Separate files such as `campaign_il7ra.yaml`
can coexist in one checkout.

GPU 0 is reserved for the local Qwen server. The other GPUs are campaign
workers. Thus, `--gpus 2`, `4` and `8` create one, three and seven workers. The
launcher currently requests all GPUs on one node.

The generated YAML also exposes enabled design families, seed, controller policy
and optional backend overrides. Most users can leave backend paths `null`; the
CLI loads installed locations from `.env` and `.env.assets`. YAML paths are
resolved relative to the YAML file.

Registered targets obtain their target constraint and PDB from the repository
and asset bundle. A custom target needs an explicit constraint JSON and PDB; see
the [custom-target guide](../examples/custom_target/README.md).

## Validate and submit

Check the complete resolved input without submitting a job:

```bash
trex check campaign_cd45.yaml
```

This verifies target identity and hashes, required checkpoint manifests, enabled
backends, the local Qwen snapshot and pinned backend revisions. Add
`--verify-checkpoint-content` for a slow full rehash of every declared model
file.

Submit using your cluster's default account and partition:

```bash
trex submit campaign_cd45.yaml
```

If your site requires scheduler identifiers, pass them explicitly:

```bash
trex submit campaign_cd45.yaml \
  --account YOUR_ACCOUNT \
  --partition YOUR_GPU_PARTITION
```

`--account` identifies the project charged for compute. `--partition` selects a
queue or node group. Valid names and defaults are site-specific. `--qos` and
repeatable `--sbatch-option` are available when required by the site.

`trex submit` validates the configuration, derives worker indices and the
Slurm GPU/time request, and invokes `slurm/T-REX.slurm`. Do not run the Slurm
file separately. The job starts and health-checks the pinned local LLM on GPU 0,
writes provenance, and starts the controller on the remaining GPUs.

The scheduler request includes startup/shutdown headroom beyond the campaign
hours. Four GPUs and `--hours 48` request 49 hours. CPU and memory defaults are
16 cores and 192 GB; pass site-specific changes with `--sbatch-option`, for
example `--sbatch-option=--mem=256G`. GPU count and time must stay in the YAML.

## Monitor and inspect outputs

A submitted run prints an archive path of this form:

```text
<output>/trex_<target>_s<seed>_<YYYYMMDD_HHMM>/<slurm-job-id>/
```

Slurm stdout and stderr are in `slurm_logs/`. LLM logs are `vllm.out` and
`vllm.err` inside the run archive.

```bash
squeue -u "$USER"
TREX_RUN_ARCHIVE=/absolute/path/to/run/archive
trex status "$TREX_RUN_ARCHIVE"
trex export "$TREX_RUN_ARCHIVE" --n 100
```

Start result review with `export/manifest.csv`; selected structures are under
`export/pdbs/`. See [outputs and analysis](outputs-and-analysis.md) for record
schemas, joins and structural-diversity analysis.

For a short molecular smoke, use `--hours 0.75` or longer. The shortest
default generator estimate is 0.5 hours, and admission also reserves a 0.1-hour
drain margin; smaller values may validate startup without launching a backend
job. Check `dispatch_records.jsonl` to confirm molecular work started.

## Resume an interrupted campaign

Keep the original YAML and submit the exact existing archive:

```bash
trex submit campaign_cd45.yaml --resume /absolute/path/to/existing/archive
```

Resume appends records and includes elapsed time already present in the archive;
it does not add another full campaign duration. Never run two controllers
against the same archive.

## Run inside an existing allocation

When you manage the GPU allocation and compatible OpenAI-style LLM server
separately, set `llm.base_url` in the YAML and run:

```bash
trex design campaign_cd45.yaml
```

This command preflights and starts the controller in the current allocation. It
does not call Slurm or start vLLM. Use `trex design campaign_cd45.yaml --dry-run` for
a read-only resolution check.

## Compatibility interface

Existing configurations can continue to use `.env` campaign variables and the
shell wrapper:

```bash
bash scripts/submit.sh --check
bash scripts/submit.sh --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION
```

The wrapper and `slurm/T-REX.slurm` remain the implementation underneath
`trex submit`; new configurations should use the YAML CLI above. The detailed
`trex campaign init/show/preflight/run/status` namespace is retained for scripts
that already use it.
