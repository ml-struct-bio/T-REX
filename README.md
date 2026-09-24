# T-REX

**Target-adaptive Rescue–Explore–eXploit** orchestrates de novo protein binder
design. Planner and Supervisor LLMs propose and prioritize work; deterministic
code validates jobs and manages generation, redesign and evaluation.

Choose a **target, GPU count, campaign duration and output directory** in one
campaign YAML. The public workflow is `trex init`, `trex check` and `trex submit`.
The Slurm launcher uses one node with at least two GPUs: one for the local LLM
and the rest for campaign jobs. Complete steps 1–2 once per installation, then
use steps 3–4 for each campaign. Run commands from the repository root.

## 1. Install the controller

Clone the repository, or enter your existing `T-REX` checkout. Use Python
3.10–3.12 for the controller. Most GPU profiles use Python 3.12.13;
BindCraft uses a separate Python 3.10.20 environment.

```bash
git clone https://github.com/ml-struct-bio/T-REX.git
cd T-REX
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[assets]'
python -m pip check
```

This installs the campaign controller and analysis commands. You can immediately
run the [CPU archive-analysis example](examples/analysis_demo/README.md).

## 2. Prepare GPU environments and checkpoints

Follow [installation](docs/installation.md) to create the separate local-LLM,
Complexa/AF2/ProteinMPNN, BindCraft and BoltzGen environments and install
Foldseek/MMseqs2. Keep the checkouts, environments and their base Python
interpreters accessible from compute nodes. The guide includes dependency and
GPU checks and distinguishes installation profiles from the study inventory.

After obtaining the backend checkouts, prepare the pinned checkpoints and target
structures (about 52 GB; see [assets](docs/assets.md) for component downloads):

```bash
python scripts/manage_assets.py fetch --root ../T-REX-assets \
  --drive-url 'https://drive.google.com/file/d/1orGqGyNFJpxUdfypbqFfs6W0XvlDJDwG/view?usp=sharing'
python scripts/manage_assets.py verify --root ../T-REX-assets
python scripts/manage_assets.py configure \
  --root ../T-REX-assets \
  --complexa-repo ./external/Proteina-Complexa \
  --community-repo ./external/Proteina-Complexa \
  --bindcraft-repo ./external/BindCraft \
  --env-out .env.assets
```

`configure` links the verified weights into the backend trees and writes
`.env.assets` in the repository root. The campaign configuration loads this file
automatically; keep it and the asset directory in place. It supplies the CD45 PDB at
`targets/bindcraft_targets/CD45.pdb`; the matching constraints are already in
[config/targets/cd45.json](config/targets/cd45.json).

## 3. Create one campaign configuration

Create the installation profile once. `.env` records backend environments and
executable paths; `.env.assets` was generated in step 2. Check the paths in
`.env` after copying the template.

```bash
cp .env.example .env
```

Then create the CD45 campaign YAML. This command records the four choices that
usually change between campaigns:

```bash
trex init campaign_cd45.yaml \
  --target cd45 \
  --gpus 4 \
  --hours 48 \
  --output /absolute/path/to/outputs
```

The first argument is the YAML file to create. Its name and location are your
choice; target-specific names such as `campaign_cd45.yaml` and
`campaign_il7ra.yaml` make multiple campaigns easier to distinguish. Pass that
same path to `trex check`, `trex submit` or `trex design` below.

`--gpus` is the total number requested on one node. GPU 0 serves the local LLM;
the remaining GPUs are workers, so `2`, `4` and `8` provide one, three and seven
workers. `--hours` is the controller campaign duration. Slurm adds bounded
startup and shutdown headroom to its allocation request. For a short molecular
smoke, use `--hours 0.75` or longer; smaller values may validate startup without
admitting a backend job.

Registered targets resolve their constraint JSON and PDB automatically from the
repository and `.env.assets`. The study labels are `cd45`, `betv1`, `cbago`,
`her2aav`, `sc2rbd`, `pdl1` and `il7ra` (main benchmark). For an unregistered target, see the
[custom-target guide](examples/custom_target/README.md).

The three configuration files have distinct roles:

| File | Role |
| --- | --- |
| `campaign_<target>.yaml` | Target, GPU workers, duration, output, policy and enabled design families |
| `.env` | Installed controller/backend executables and site-specific runtime settings |
| `.env.assets` | Verified checkpoint and target paths written by the asset manager |

## 4. Check and submit

Validate target identity, model files, enabled backends and pinned backend
revisions without submitting a job:

```bash
trex check campaign_cd45.yaml
```

Submit from a Slurm login/submission node:

```bash
trex submit campaign_cd45.yaml
```

If your cluster requires an explicit project account or GPU partition:

```bash
trex submit campaign_cd45.yaml \
  --account YOUR_ACCOUNT \
  --partition YOUR_GPU_PARTITION
```

`--account` is the cluster project charged for compute and `--partition` is its
GPU queue. Their valid names come from your cluster. The command derives the
Slurm GPU/time request from the campaign YAML, repeats validation and invokes
[slurm/T-REX.slurm](slurm/T-REX.slurm); do not run the Slurm file separately.

When you already have a GPU allocation and a compatible OpenAI-style LLM server,
set its endpoint in the YAML and run:

```bash
trex design campaign_cd45.yaml
```

`trex design` runs in the current allocation. `trex submit` obtains a new Slurm
allocation and starts the pinned local Qwen server automatically. The legacy
`scripts/submit.sh` interface remains available for existing `.env` workflows;
new users can keep all campaign choices in the YAML. See the
[run guide](docs/slurm.md) for monitoring, resume and scheduler details.

## 5. Inspect the archive and export results

For CD45, a run has the following layout. JSONL files contain one JSON object
per line and appear when that record type is first written. Updated hypotheses
append a new version with the same ID; previous lines remain in the archive.

```text
outputs/
  trex_cd45_s0_<timestamp>/<slurm-job-id>/
    result_records.jsonl         # ResultRecord: outputs, measurements, costs, failures
    evidence_summaries.jsonl     # EvidenceSummary: evidence at each planning update
    hypothesis_cards.jsonl      # HypothesisCard: proposals and subsequent feedback
    action_candidates.jsonl     # Checked candidate jobs
    supervisor_decisions.jsonl  # Rankings and Rescue/Explore/eXploit allocations
    llm_call_records.jsonl      # Model-call metadata, validation, latency, token use
    launch_decisions.jsonl      # Queue-admission approvals/rejections
    dispatch_records.jsonl      # Confirmed worker starts, deferrals and failures
    panel_selections.jsonl      # Panel records, when emitted
    campaign_input_<sha>.yaml   # Exact submitted campaign configuration
    campaign_resolved_<sha>.json # Absolute paths and effective controller input
    run_provenance.json          # Actual source, software, inputs and model identities
    controller_checkpoint.json   # Resume state
    worker_outputs/             # Per-job logs and backend-specific outputs
    vllm.out
    vllm.err
    export/                     # Created by trex export
      manifest.csv
      manifest.json
      pdbs/
```

The job prints its archive path, under the configured output directory. Use it
in these commands (the target ID for CD45 is `05_CD45`):

```bash
TREX_RUN_ARCHIVE=/absolute/path/to/outputs/trex_cd45_s0_TIMESTAMP/JOB_ID
trex status "$TREX_RUN_ARCHIVE"
trex export "$TREX_RUN_ARCHIVE" --n 100
trex-analyze validate --archive-root "$TREX_RUN_ARCHIVE"
trex-analyze summary --archive-root "$TREX_RUN_ARCHIVE"
trex-analyze trace --archive-root "$TREX_RUN_ARCHIVE" --limit 10
```

`export/manifest.csv` contains measurements and `export/pdbs/` contains the
exported structures. Complexa writes its native structures and score tables
under `$TREX_COMPLEXA_REPO/inference/`; its worker log is in the archive.
Result records reference those source paths. Preserve both locations, or export
the structures you need before removing backend outputs.
See [outputs and analysis](docs/outputs-and-analysis.md) for JSON examples, record
joins, optional streams and structural-diversity selection. The latest online
summary is distinct from the paper's recomputed final SU endpoint.

To integrate another design method, see [adding a backend](CONTRIBUTING.md#add-a-backend).
See [reproducibility](docs/reproducibility.md) for run identities, configuration
semantics and the scope of the published software.

See [CITATION.cff](CITATION.cff) for citation. T-REX uses the [MIT License](LICENSE);
[upstream software and model citations](docs/citations.md) are listed separately.
External software, weights and data retain their own licenses.
