# T-REX

**Target-adaptive Rescue-Explore-eXploit** orchestrates de novo protein-binder
design. Planner and Supervisor LLMs propose and prioritize jobs; deterministic
code validates those jobs and manages generation, redesign and evaluation.

A campaign needs four user choices: a target, the number of GPUs on one node,
the campaign duration and an output directory. The normal workflow is:

```text
trex setup -> trex init -> trex check -> trex submit
```

## Requirements

The verified installation profile targets Linux x86_64 and NVIDIA H100 GPUs.
Run the setup on storage visible from both Slurm login and compute nodes. Before
starting, provide:

- Git, Git LFS, CMake, C/C++ and Rust compilers, zlib/bzip2 development
  libraries, Bash and curl;
- an NVIDIA driver compatible with the locked PyTorch CUDA runtimes;
- Slurm commands such as `sbatch`, `squeue` and `sacct` for submission;
- Internet access during setup, or a locally downloaded asset ZIP;
- at least 110 GB for the asset ZIP and extracted checkpoints, plus space for
  the separate Python environments and campaign outputs.

T-REX uses separate environments because the verified Complexa, BindCraft,
BoltzGen and vLLM stacks require incompatible Python, JAX and PyTorch versions.
The setup command creates and configures these environments automatically.
PyRosetta is downloaded from its official wheel index for BindCraft and remains
subject to the upstream PyRosetta license.

## 1. Install T-REX, backends and checkpoints

Start in a fresh shell, clone the repository and install the pinned `uv`
standalone binary. This installer does not use Python or an existing virtual
environment. Then run one setup command from the repository root:

```bash
# Leave any virtual environment from an older checkout.
deactivate 2>/dev/null || true
hash -r

git clone https://github.com/ml-struct-bio/T-REX.git
cd T-REX

curl -LsSf https://astral.sh/uv/0.11.1/install.sh | \
  env UV_NO_MODIFY_PATH=1 sh
export PATH="$HOME/.local/bin:$PATH"
uv --version

uv run --locked --python 3.12.13 --extra assets trex setup \
  --asset-root ../T-REX-assets
```

If the prompt still shows an environment from an older T-REX checkout, run
`deactivate` or open a new shell before starting. A moved or deleted active
`.venv` can otherwise leave `python` pointing at a path that no longer exists.

The final command creates the controller environment and then performs the full
installation:

1. checks out the recorded Complexa, BindCraft and BoltzGen commits;
2. downloads and SHA256-verifies `T-REX-assets.zip` from the
   [published Google Drive file](https://drive.google.com/file/d/1QkXn7AoHD-pasiHoIx5o08TlTrWjrqKx/view?usp=share_link);
3. links the verified checkpoints and seven benchmark target PDBs into the
   backend checkouts;
4. creates the locked vLLM/controller, Complexa/AF2/ProteinMPNN, BindCraft and
   BoltzGen environments;
5. builds the recorded Foldseek and MMseqs2 commits;
6. writes Complexa's runtime configuration, `.env.assets` and the root `.env`.

The download is about 44.8 GB as a ZIP and about 52 GB after extraction. If
Google Drive is rate-limited, download `T-REX-assets.zip` in a browser and use:

```bash
uv run --locked --python 3.12.13 --extra assets trex setup \
  --asset-root ../T-REX-assets \
  --asset-zip /absolute/path/to/T-REX-assets.zip
```

`trex setup` is resumable. Rerun the same command after a network or
login-session interruption. It verifies completed stages and continues from the
first incomplete stage. It refuses to overwrite an unrelated backend checkout,
Python environment or manually created configuration file.

The installation creates these paths:

```text
T-REX/
  .venv/                              # lightweight public CLI
  .venv-serving/                      # local LLM server and campaign controller
  .env                                # backend executable profile
  .env.assets                         # verified asset paths
  external/
    Proteina-Complexa/.venv/          # Complexa, AF2 and ProteinMPNN
    BindCraft/.venv/                  # BindCraft and PyRosetta
    BoltzGen/.venv/                   # BoltzGen
    Foldseek/                          # recorded source revision
    MMseqs2/                           # recorded source revision
../T-REX-assets/
  checkpoints/
  targets/
  manifest.json
```

Activate the CLI environment for subsequent commands:

```bash
source .venv/bin/activate
```

The installed command is `trex`. You can also prefix any command with
`uv run --locked --extra assets` instead of activating `.venv`.

## 2. Create a campaign

This example creates a 48-hour CD45 campaign using four GPUs:

```bash
trex init campaign_cd45.yaml \
  --target cd45 \
  --gpus 4 \
  --hours 48 \
  --output "$PWD/outputs"
```

The first argument is the YAML file to create. Its name and location are your
choice, so `campaign_cd45.yaml`, `campaign_il7ra.yaml` and similar names can
coexist.

`--gpus` is the total number of GPUs requested on one node. GPU 0 serves the
local LLM; the remaining GPUs run molecular-design jobs. Thus `--gpus 4`
provides three worker GPUs. `--hours` is the controller campaign duration.
`--output` may be any absolute or shell-expanded path visible from compute
nodes; each submission creates a new timestamped run below it.

The registered main-benchmark target labels are:

```text
cd45  betv1  cbago  her2aav  sc2rbd  pdl1  il7ra
```

Each label resolves the exact constraint JSON and verified target PDB supplied
with the release. See [the custom-target example](examples/custom_target/README.md)
to use another target.

A campaign YAML contains the target, workers, time budget, output directory,
policy and enabled design families. `.env` contains executable paths, and
`.env.assets` contains checkpoint and target paths. New users normally edit
only the campaign YAML.

## 3. Validate before using GPUs

Run the fail-closed preflight:

```bash
trex check campaign_cd45.yaml
```

This checks target identity and hotspots, checkpoint files, the serving
interpreter, all enabled backend executables, Git revisions, the BindCraft
production patch, and the recorded Foldseek/MMseqs2 versions. Submission stops
if a required check fails.

To rehash every large model file as well as checking its path and size:

```bash
trex check campaign_cd45.yaml --verify-checkpoint-content
```

## 4. Submit to Slurm

From a Slurm login node:

```bash
trex submit campaign_cd45.yaml
```

Some clusters infer the account and GPU partition from your user defaults. If
your site requires them, supply both explicitly:

```bash
trex submit campaign_cd45.yaml \
  --account YOUR_ACCOUNT \
  --partition YOUR_GPU_PARTITION
```

`--account` is the project charged for compute. `--partition` is the GPU
queue. Their valid values are cluster-specific; `sacctmgr show user "$USER"`
and `sinfo` often display them, or your cluster documentation will list them.

`trex submit` derives GPU count and wall time from the campaign YAML, repeats
preflight, submits [slurm/T-REX.slurm](slurm/T-REX.slurm), starts the local Qwen
server on GPU 0 and starts workers on the remaining GPUs. Do not invoke the
Slurm file separately.

For a short end-to-end molecular smoke, make a separate configuration:

```bash
trex init campaign_cd45_smoke.yaml \
  --target cd45 \
  --gpus 4 \
  --hours 0.75 \
  --output "$PWD/smoke-outputs"

trex check campaign_cd45_smoke.yaml
trex submit campaign_cd45_smoke.yaml
```

A duration below about 0.5 hours may confirm startup but end before a molecular
backend job is admitted. Installation checks and a short smoke establish
operability; they do not reproduce a 48-hour benchmark result.

If you already hold a GPU allocation and provide a compatible OpenAI-style LLM
endpoint in the YAML, run the controller directly:

```bash
trex design campaign_cd45.yaml
```

## 5. Monitor, resume and inspect outputs

Use standard Slurm commands to monitor the allocation:

```bash
squeue -u "$USER"
sacct -j JOB_ID --format=JobID,State,Elapsed,AllocTRES,ExitCode
```

Find the created archive below the YAML's `run.archive_root`, then inspect it:

```bash
trex status /absolute/path/to/one/archive
```

Resume an interrupted archive without creating a new campaign history:

```bash
trex submit campaign_cd45.yaml \
  --resume /absolute/path/to/one/archive
```

A CD45 archive has this structure:

```text
outputs/
  trex_cd45_s0_<timestamp>/<slurm-job-id>/
    result_records.jsonl
    evidence_summaries.jsonl
    hypothesis_cards.jsonl
    action_candidates.jsonl
    supervisor_decisions.jsonl
    llm_call_records.jsonl
    launch_decisions.jsonl
    dispatch_records.jsonl
    panel_selections.jsonl
    campaign_input_<sha>.yaml
    campaign_resolved_<sha>.json
    run_provenance.json
    controller_checkpoint.json
    worker_outputs/
    vllm.out
    vllm.err
```

JSONL files contain one JSON object per line and appear when that record type is
first emitted. Hypothesis updates append new versions with the same ID; prior
records remain in the archive. `run_provenance.json` records the source
revision, target and model hashes, backend revisions and runtime versions.

Export a ranked set of structures and a CSV/JSON manifest with:

```bash
trex export /absolute/path/to/one/archive --n 100
```

The export is written below `archive/export/`. A CPU-only archive-analysis
example is available in [examples/analysis_demo](examples/analysis_demo/README.md).

## Reproducibility scope

The default campaign enables all six generation families plus ProteinMPNN
redesign and AF2 evaluation. The public configuration preserves the target
inputs, action space, controller behavior, environment locks, source revisions
and checkpoint manifests used for the study. See
[reproducibility details](docs/reproducibility.md) for the distinction between
the historical environment inventory and the compatible clean-install profiles.

Backend software and model assets retain their upstream licenses. Official
sources and research references are listed in
[docs/citations.md](docs/citations.md). Cite T-REX with [CITATION.cff](CITATION.cff).

For development and release checks, see [CONTRIBUTING.md](CONTRIBUTING.md).
