# Installation

## Controller and analysis package

Use Python 3.10, 3.11 or 3.12. From a downloaded or cloned T-REX repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pip check
trex --help
```

The editable install links commands to this checkout, keeping executed source
identity consistent with `backends.repo_root`. Keep the checkout in place.
A regular wheel install also supports archive analysis; use the checkout install
for the campaign and Slurm instructions in this guide.

This installs the controller and archive-analysis commands. It does not install
scientific backends, model weights or vLLM. Try the
[CPU analysis example](../examples/analysis_demo/README.md) before setting up GPUs.
For development and the complete release gate, see [development](development.md).

## Requirements for a campaign

A complete campaign needs the external environments and assets below. Keep
backend environments separate: the controller launches their configured
executables. Install the dependencies required by your enabled action families;
the paper profile uses all eight families.

| Component | Required for |
| --- | --- |
| Complexa generation environment and weights | Four Complexa generation search variants |
| BindCraft environment and assets | BindCraft generation |
| BoltzGen environment and checkpoints | BoltzGen generation |
| AF2 / ProteinMPNN asset environment | Standardized evaluation / sequence redesign |
| Foldseek and MMseqs2 | Structural and sequence clustering |
| LLM endpoint and local model identity manifest | Planner and Supervisor in the YAML campaign workflow |
| Target structure and constraints | Every campaign |

Generation outputs can require standardized AF2 evaluation, so choosing a
generation backend does not necessarily remove the AF2 dependency. The strict
preflight resolves these dependencies from the enabled families.

Source checkouts provide the Slurm templates and examples. Wheels include
target constraints, reproducibility manifests, the tested BindCraft patch and
the canonical prompt snapshot. Both installation modes require external
backend environments, target structures and model/checkpoint assets.

For Slurm, the virtual environment **and its base Python executable** must be
accessible on compute nodes. Use a cluster module or Python installed on shared
storage; a venv pointing to a login node's `/tmp` will not work on compute nodes.

## Local vLLM controller environment


For the exact local-vLLM environment used in production:

```bash
python -m venv .venv-controller
source .venv-controller/bin/activate
python -m pip install --upgrade pip
python -m pip install -r config/trex/controller_requirements.lock.txt
python -m pip install -e .
```

The lock captures resolved package versions but is not a cross-platform wheel
lock. It was tested on Linux, Python 3.12.13, CUDA-enabled H100 nodes, vLLM
0.19.0, and PyTorch 2.10.0. Rebuilding on a different CUDA/driver stack can
require compatible wheels and is a distinct environment.

## Install pinned scientific backends

The tested commits are machine-readable in
`config/reproducibility/production_stack.json`.

```bash
mkdir -p external

git clone https://github.com/NVIDIA-Digital-Bio/Proteina-Complexa \
  external/Proteina-Complexa
git -C external/Proteina-Complexa checkout \
  5ae24b055d828918296f2aad63616b1f4cc0e491

# Separate tested community-model tree for AF2/ProteinMPNN assets.
git clone https://github.com/NVIDIA-Digital-Bio/Proteina-Complexa \
  external/Proteina-Complexa-community
git -C external/Proteina-Complexa-community checkout \
  d323517efbffe3ea280e4e3d0442b0257a37b5a8

git clone https://github.com/martinpacesa/BindCraft external/BindCraft
git -C external/BindCraft checkout \
  b971db42ba6e091afab63ccb30ae02215150a990
git -C external/BindCraft apply \
  "$(pwd)/external/patches/bindcraft-production.patch"

git clone https://github.com/HannesStark/boltzgen external/BoltzGen
git -C external/BoltzGen checkout \
  31d9d9b9c72245b4ed6fe8742d6fbf4e1a3552a0
```

Follow each project's official installation instructions at that revision.
T-ReX expects:

- Proteina-Complexa's `env.sh`, generation checkpoints, and executable Python;
- a legacy/community-model asset tree containing AF2 multimer parameters,
  ColabDesign, and ProteinMPNN;
- BindCraft's executable environment plus DSSP and DAlphaBall;
- the `boltzgen` executable and checkpoint cache;
- Foldseek; and
- MMseqs2 for sequence-diversity feedback and final panel analysis.

The AF2 and ProteinMPNN assets may reside under a second
`TREX_LEGACY_COMPLEXA_REPO`; the generation checkout remains
`TREX_COMPLEXA_REPO`. This split matches the tested deployment and avoids
requiring one Python environment to own all third-party dependencies.

The exact AF2 and ProteinMPNN asset trees are recorded in
`config/reproducibility/af2_parameters_manifest.json` and
`config/reproducibility/proteinmpnn_weights_manifest.json`. The files remain
under their upstream licenses and are not redistributed.

## Install Foldseek and MMseqs2

Use the official binaries or a cluster module. Prefer exact executable paths:

```bash
export TREX_FOLDSEEK_BIN=/absolute/path/to/foldseek
export TREX_MMSEQS_BIN=/absolute/path/to/mmseqs
```

The tested versions are:

- Foldseek `8dc75c74ad0eddab73cfd905963d13bf74dc012b`
- MMseqs2 `76da68ad7577378410c075049e18666fcc94f8d1`

On module-based clusters the Slurm template also accepts
`TREX_FOLDSEEK_MODULE` and `TREX_MMSEQS_MODULE`.

## Stage target structures

Target PDB files are not duplicated in this repository. Point
`TREX_TARGET_ASSET_ROOT` at a tree matching `config/targets/registry.json`.
The exact registered structures are checked against
`config/targets/assets.sha256.json`.

```bash
export TREX_TARGET_ASSET_ROOT=/path/to/Proteina-Complexa/assets/target_data
trex-validate --target cd45 --asset-root "$TREX_TARGET_ASSET_ROOT"
```

Do not substitute a different crop, chain assignment, repaired structure, or
PDB revision while retaining the same benchmark target name. Register it as a
new target and report the new SHA256.

## Stage the LLM

The checked-in manifest describes `Qwen/Qwen3.6-27B-FP8` at upstream revision
`ec4160bf26124fa57e6451d070ee0c459a36d5b7`. Its content digest is:

```text
77409ac00f81e48d29b9f079ab448044a5ec1cdd91dfdb50b3035c197f065722
```

Set:

```bash
export TREX_QWEN_MODEL_PATH=/path/to/Qwen3.6-27B-FP8
export TREX_MODEL_MANIFEST="$PWD/config/trex/qwen3_6_27b_fp8_model_manifest.json"
```

Startup validates manifest integrity plus every declared file name and byte size
against the full content-hash manifest. To generate a manifest for a different
model:

```bash
trex-provenance hash-model \
  --model-path /path/to/model \
  --out /path/to/model-manifest.json
```

Revalidate every model byte for a release audit with:

```bash
trex-provenance model-digest \
  --model-path "$TREX_QWEN_MODEL_PATH" \
  --manifest "$TREX_MODEL_MANIFEST" \
  --full-content
```

Changing the model or manifest defines a distinct experiment.

## Configure paths

```bash
cp .env.example .env
# Edit all paths.
set -a
source .env
set +a
```

`.env` is not parsed automatically. The `set -a`/`set +a` pair exports its
assignments to the controller and worker subprocesses.

## Fail-closed preflight

```bash
trex-validate \
  --target cd45 \
  --asset-root "$TREX_TARGET_ASSET_ROOT" \
  --enabled-families "$TREX_ENABLED_FAMILIES" \
  --require-backends \
  --require-model \
  --verify-backend-revisions
```

Add `--verify-checkpoint-content` for the slower release audit that rehashes the
Qwen, AF2, and ProteinMPNN files. The reference
`slurm/publication_preflight.slurm` enables this mode.

The preflight accepts a separate `TREX_TEST_PYTHON` with `.[dev]` installed for
pytest and deterministic replays. `TREX_VALIDATION_PYTHON` may remain the lean
production controller environment; test-only packages do not need to be added
to it.

The command performs no GPU work. It validates:

- target registry/config/PDB identity, chains, hotspots, and SHA256;
- only the backends required by enabled families;
- generation checkpoints, AF2 parameters, and ProteinMPNN weights;
- Foldseek and optional MMseqs2 executables;
- the local model manifest; and
- pinned backend Git revisions when requested.

An enabled-family failure is fatal. An unavailable optional MMseqs2 binary is a
warning because it disables sequence-diversity evidence without changing the
strict SU endpoint.

## Slurm smoke and launch

```bash
bash -n slurm/trex_per_target_node.slurm
mkdir -p slurm_logs
export TARGET=cd45
sbatch --export=ALL slurm/trex_per_target_node.slurm
```

The template deliberately omits site-specific account, partition, QOS, and
reservation directives. Supply them at submission. The default layout is the
paper-reference four-GPU job: GPU 0 hosts the local LLM and GPUs 1-3 are worker
slots. To scale the same controller to a larger node, request more GPUs from
Slurm, set `TREX_WORKER_GPUS` to the worker indices, and set
`TREX_REQUIRE_THREE_WORKERS=0`:

```bash
export TREX_WORKER_GPUS=1,2,3,4,5,6,7
export TREX_REQUIRE_THREE_WORKERS=0
export TREX_CHARGED_GPUS=8  # optional raw audit metadata only
sbatch --gres=gpu:8 --export=ALL slurm/trex_per_target_node.slurm
```

Always export comma-separated values in the shell and submit with
`--export=ALL`. Do not write a comma-valued `TREX_WORKER_GPUS` or
`TREX_ENABLED_FAMILIES` assignment inside `sbatch --export=...`; Slurm parses
those commas as variable separators and can silently retain only the first
GPU or family.

### Optional smoke checks

Use subshells for optional examples so their overrides do not affect the
subsequent production run. Create `slurm_logs/` before submitting any job.

Before spending a full campaign budget, verify a scaled Slurm allocation and
per-worker GPU mapping without loading an LLM or scientific backend:

```bash
(
  mkdir -p slurm_logs
  export TREX_WORKER_GPUS=1,2,3,4,5,6,7
  export TREX_TEST_PYTHON=/path/to/trex-dev/bin/python
  sbatch --gres=gpu:8 --export=ALL slurm/multigpu_smoke.slurm
)
```

This bounded smoke checks all requested GPU indices, unique device UUIDs,
worker-wall scaling, event-driven slot behavior, and Selector behavior. It does
not claim an end-to-end molecular result; use a short target campaign after it
when validating a new hardware/backend stack.

By default the smoke expects every allocated GPU except GPU 0 to be a worker;
this catches a comma-truncated `TREX_WORKER_GPUS`. Set
`TREX_EXPECTED_WORKER_COUNT` only when intentionally testing a smaller subset.

For the follow-up end-to-end stack check, use the production launcher with a
bounded controller budget:

```bash
(
  mkdir -p slurm_logs
  export TARGET=cd45
  export TREX_MAX_WALL_H=0.65
  export TREX_ENABLED_FAMILIES=complexa_beam,complexa_best_of_n,complexa_fk_steering,structure_refilter
  sbatch --time=01:00:00 --export=ALL slurm/trex_per_target_node.slurm
)
```

This assumes the site-specific paths above pass strict preflight. Startup
consumes scheduler time and shutdown waits are sequential, so a short allocation
does not guarantee that all work finishes. Inspect process exits and archive
validation before interpreting the result. This bounded stack check does not
establish reproduction of a full campaign.

The reference full campaign uses 48 cumulative controller hours and a 49-hour
Slurm reservation, with the reporting denominator kept at 144 worker GPU-hours
for three worker slots. See the
[timing contract](reproducibility.md#execution-timing-and-shutdown).

## Installation boundaries

The repository cannot legally or practically contain AF2/model weights, backend
checkpoints, or every official backend environment. A checkout is therefore
not a one-command molecular-design appliance. The complete, testable contract
is the combination of:

1. this source revision;
2. `production_stack.json`;
3. target and model content manifests;
4. the external licensed assets;
5. successful `trex-validate --require-backends --require-model`; and
6. the per-run `run_provenance.json`.
