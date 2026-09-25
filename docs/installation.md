# Installation profiles and internals

The supported installation path is the one-command setup in the main
[README](../README.md#1-install-t-rex-backends-and-checkpoints):

```bash
uv run --locked --python 3.12.13 --extra assets trex setup \
  --asset-root ../T-REX-assets
```

That command performs the source checkouts, asset verification, separate
environment installs, Foldseek/MMseqs2 builds and runtime configuration
described below. The remainder of this page records the individual profiles for
audit and troubleshooting; a normal installation should not repeat them
manually.

The automatic setup keeps backend environments separate and places them inside
the checkout so the login and compute nodes use the same installation.

Use the compatible installation profiles below. Software identities and the
distinction between installation checks and benchmark reproduction are described
in [reproducibility](reproducibility.md).

Install Git and Git LFS on both login and compute nodes. Backend revision checks
invoke Git and may need its LFS filter even when weights were downloaded
separately. Also make `bash`, `curl`, and the Slurm commands available. Check
`git lfs version` inside a compute allocation as well as on the login node.

## Controller and LLM environment

On Linux x86_64, use an NVIDIA driver compatible with PyTorch 2.10.0's CUDA
12.8 build, a CUDA toolkit (`nvcc`) and a C++ compiler. FlashInfer builds kernels
on its first run. Load your site's CUDA module or set `CUDA_HOME`; keep the
venv and its base Python on compute-node-accessible storage.

From the T-REX checkout, create a separate serving environment:

```bash
uv --version  # must report uv 0.11.1
uv venv --python 3.12.13 .venv-serving
uv pip sync --python .venv-serving/bin/python \
  config/trex/serving_requirements.lock.txt
uv pip install --python .venv-serving/bin/python --no-deps -e .
uv pip check --python .venv-serving/bin/python
```

This profile pins 179 dependencies, including vLLM 0.19.0, PyTorch 2.10.0,
Transformers 4.57.6 and NumPy 2.2.6. It is the default in `.env.example`.
The launcher activates the environment so installed tools such as `ninja` are
on `PATH`.

The [profile manifest](../config/reproducibility/serving_environment.json)
records the Python version, lock hash and validation scope.

## Backend source and environments

From the T-REX repository root, fetch the three pinned backend checkouts:

```bash
mkdir -p external
export GIT_LFS_SKIP_SMUDGE=1
git clone https://github.com/NVIDIA-BioNeMo/Proteina-Complexa external/Proteina-Complexa
git -C external/Proteina-Complexa checkout 5ae24b055d828918296f2aad63616b1f4cc0e491
git clone https://github.com/martinpacesa/BindCraft external/BindCraft
git -C external/BindCraft checkout b971db42ba6e091afab63ccb30ae02215150a990
git -C external/BindCraft apply "$PWD/external/patches/bindcraft-production.patch"
git clone https://github.com/HannesStark/boltzgen external/BoltzGen
git -C external/BoltzGen checkout 31d9d9b9c72245b4ed6fe8742d6fbf4e1a3552a0
unset GIT_LFS_SKIP_SMUDGE
```

The checkpoints are supplied by the asset bundle, so these commands skip
duplicate Git LFS downloads. The BindCraft patch sets the executable permissions
used in the study.

Then install each backend in its own environment:

| Backend | Setup and path to configure |
| --- | --- |
| [Complexa](https://github.com/NVIDIA-BioNeMo/Proteina-Complexa/blob/5ae24b055d828918296f2aad63616b1f4cc0e491/README.md#installation) | Use the compatible installation profile below for generation, AF2 and ProteinMPNN. Set `TREX_COMPLEXA_REPO` and `TREX_COMPLEXA_PYTHON`. |
| [BindCraft](https://github.com/martinpacesa/BindCraft/blob/b971db42ba6e091afab63ccb30ae02215150a990/README.md#installation) | Use the separate Python 3.10 profile below, or the upstream conda installer. Set `TREX_BINDCRAFT_REPO` and `TREX_BINDCRAFT_ENV` to the resulting environment prefix. |
| [BoltzGen](https://github.com/HannesStark/boltzgen/blob/31d9d9b9c72245b4ed6fe8742d6fbf4e1a3552a0/README.md#installation) | Use the separate Python 3.12 profile below. Set `TREX_BOLTZGEN_REPO`, `TREX_BOLTZGEN_BIN` and `TREX_BOLTZGEN_CACHE`. Stage weights in that cache before using offline compute nodes. |

Use this same public Complexa checkout for AF2/ProteinMPNN. The asset
configuration command in the README writes both checkout paths to `.env.assets`;
the campaign's `.env` loads them when you submit.

Revision preflight verifies the `community_models/colabdesign/` and
`community_models/ProteinMPNN/` source trees and rejects tracked changes in them.
The Python selected by `TREX_COMPLEXA_PYTHON` runs generation, AF2 evaluation
and ProteinMPNN; it must support all three.

Backend setup may require additional upstream assets, such as BindCraft's
DSSP, DAlphaBall and PyRosetta. Complete those installations; the T-REX path
validator does not replace a backend's installation checks.

## Complexa, AF2 and ProteinMPNN environment

From the T-REX root, install the pinned Linux x86_64 profile into a new directory:

```bash
python scripts/setup_complexa_env.py \
  --repo external/Proteina-Complexa --env external/Proteina-Complexa/.venv
```

This requires `uv==0.11.1`, installed above. It checks the backend revision,
applies a dependency-metadata patch, installs the locked dependencies and local
ColabDesign, and checks dependency compatibility, imports and the Complexa CLI.
It refuses to replace an existing environment that it did not create. Select a
new `--env` path if you already have an upstream environment, and use that path
for `TREX_COMPLEXA_PYTHON` and Complexa's `UV_VENV` setting.

The [profile](../config/reproducibility/complexa_environment.json) records source
revisions, dependency metadata, lock hashes and validation scope.

Create Complexa's runtime configuration template:

```bash
(
  cd external/Proteina-Complexa
  source .venv/bin/activate
  complexa init
)
```

Edit `external/Proteina-Complexa/.env` before generating `env.sh`.
Set `LOCAL_CODE_PATH` to its absolute checkout path,
`LOCAL_DATA_PATH` to the absolute `T-REX-assets/targets` directory and
`LOCAL_CHECKPOINT_PATH` to its `ckpts` directory. For the uv runtime, configure
`UV_FOLDSEEK_EXEC`, `UV_MMSEQS_EXEC` and `UV_DSSP_EXEC` with absolute executable
paths. These generate the corresponding runtime variables. `UV_SC_EXEC` is
needed if you enable optional shape-complementarity evaluation; the default
T-REX generation configuration uses the AF2 reward without that optional
bioinformatics reward. Install the tools described in
[local LLM and clustering tools](#local-llm-and-clustering-tools) and use their
actual executable paths here. T-REX's root `.env` holds the campaign settings;
this backend `.env` holds Complexa's runtime paths.

After saving the backend `.env`, generate `env.sh`:

```bash
(
  cd external/Proteina-Complexa
  source .venv/bin/activate
  complexa init uv
)
```

T-REX sources this `env.sh` before generation. If you chose a different
Complexa environment directory, use its activation script in both commands.

After [asset configuration](assets.md#connect-assets-to-backend-checkouts), run
this inside a GPU allocation with at least 64 GB host RAM:

```bash
external/Proteina-Complexa/.venv/bin/python scripts/check_complexa_env.py \
  --repo external/Proteina-Complexa --gpu \
  --checkpoint-root ../T-REX-assets/checkpoints --report complexa-check.json
```

This checks PyTorch and JAX GPU execution, companion libraries, AF2 parameter
loading, and loading the Complexa generator and autoencoder onto the GPU. It
does not generate a binder or validate an entire campaign. Omit `--gpu` and
`--checkpoint-root` for dependency/import checks on a login node.

## BindCraft environment

From the T-REX root, use a new environment path. PyRosetta is separately licensed;
ensure that your use is covered by its upstream license before downloading it.
The command below uses the official quarterly release wheel index.

```bash
uv venv --python 3.10.20 external/BindCraft/.venv
uv pip sync --python external/BindCraft/.venv/bin/python \
  config/trex/bindcraft_requirements.lock.txt \
  --find-links https://west.rosettacommons.org/pyrosetta/quarterly/release.cxx11thread.serialization
uv pip check --python external/BindCraft/.venv/bin/python
```

This profile pins 74 packages, including NumPy 1.26.4, JAX 0.6.0, Flax 0.9.0,
Optax 0.2.8 and the recorded PyRosetta quarterly build. ColabDesign and PDBFixer
are pinned to Git commits. The checkout supplies DSSP and DAlphaBall; verify
that their shared-library dependencies are available on your compute nodes.
Asset configuration supplies the AF2 parameters separately.

T-REX isolates backend CUDA libraries from the serving toolkit. When invoking
BindCraft directly, use its environment's library path instead of inheriting
another toolkit's `LD_LIBRARY_PATH`.

Set `TREX_BINDCRAFT_ENV` to the absolute path of `external/BindCraft/.venv`.
The upstream `install_bindcraft.sh` instead creates a conda environment named
`BindCraft`; do not run it over an existing environment you need to preserve.
If you use that installer, pin ColabDesign as described below.

## BoltzGen environment

From the T-REX root:

```bash
uv venv --python 3.12.13 external/BoltzGen/.venv
uv pip sync --python external/BoltzGen/.venv/bin/python \
  config/trex/boltzgen_requirements.lock.txt
uv pip install --python external/BoltzGen/.venv/bin/python \
  --no-deps -e external/BoltzGen
uv pip check --python external/BoltzGen/.venv/bin/python
```

This compatible profile pins 101 dependencies plus the pinned local backend.
It includes PyTorch 2.14.0 with CUDA 13 runtime libraries and NumPy 2.0.2;
check driver compatibility on your GPU nodes. Set `TREX_BOLTZGEN_BIN` to the absolute
`external/BoltzGen/.venv/bin/boltzgen` path. CUDA arithmetic alone does not
validate BoltzGen model inference.

## Check each backend environment

From the T-REX root, check the environments created above:

```bash
uv pip check --python external/Proteina-Complexa/.venv/bin/python
uv pip check --python external/BindCraft/.venv/bin/python
uv pip check --python external/BoltzGen/.venv/bin/python
```

If you installed an environment elsewhere, replace its interpreter path with
the one you selected. These commands can run before the campaign `.env` exists.

BindCraft's upstream installer does not pin ColabDesign. In the activated
BindCraft environment, install the recorded package source and verify its weights:

```bash
python -m pip install 'colabdesign @ git+https://github.com/sokrypton/ColabDesign.git@e31a56fe1d9b4de25c8697f3a28b75892941cc72'
python -m pip check
python /absolute/path/to/T-REX/scripts/manage_assets.py verify-bindcraft \
  --python /absolute/path/to/BindCraft/environment/bin/python
```

The package supplies its own converted ProteinMPNN weights (`.pkl`) and a small
AF2 template array; the asset bundle retains matching copies. They are distinct
from the separate PyTorch ProteinMPNN weights (`.pt`). The verifier checks package
metadata and file hashes without loading models. Backend inference still needs
its own smoke test after dependency changes. Official repositories and research
references are in [upstream citations](citations.md).

## Checkpoint locations

The [asset setup guide](assets.md) collects the exact checkpoint files in one
folder and provides download, SHA256 verification and path-configuration tools:

```bash
python scripts/manage_assets.py fetch --root ../T-REX-assets
python scripts/manage_assets.py verify --root ../T-REX-assets
```

This downloads about 52.0 GB. Use `--component targets` for target structures
alone. Install the backend software separately as described above.

Run `manage_assets.py configure` as shown in the asset guide, then copy
[.env.example](../.env.example) to `.env` and set the executable paths. The launcher resolves
some checkpoints relative to their backend trees; these locations must contain
the files or symlinks to them:

| Asset | Required location |
| --- | --- |
| Complexa generator | `$TREX_COMPLEXA_REPO/ckpts/complexa.ckpt` |
| Complexa autoencoder | `$TREX_COMPLEXA_REPO/ckpts/complexa_ae.ckpt` |
| AF2 parameters | `$TREX_LEGACY_COMPLEXA_REPO/community_models/ckpts/AF2/` |
| ProteinMPNN weights | `$TREX_LEGACY_COMPLEXA_REPO/community_models/ProteinMPNN/vanilla_model_weights/` |
| BindCraft AF2 parameters | Its upstream-configured parameter directory |
| BoltzGen weights/cache | `$TREX_BOLTZGEN_CACHE` (create and populate this directory) |
| Qwen model snapshot | `$TREX_QWEN_MODEL_PATH` |

If Complexa downloads weights elsewhere, link the two files into its `ckpts/`
directory. Configure Complexa's own `.env` for its target-data, AF2 and tool
paths as well; the launcher sources its `env.sh`. T-REX's `.env` configures
the controller and does not replace backend configuration.

AF2 and ProteinMPNN file names, sizes and SHA256 hashes are recorded in
[af2_parameters_manifest.json](../config/reproducibility/af2_parameters_manifest.json)
and [proteinmpnn_weights_manifest.json](../config/reproducibility/proteinmpnn_weights_manifest.json).
The ProteinMPNN tree includes `v_48_002.pt`, `v_48_010.pt`, `v_48_020.pt` and
`v_48_030.pt`. These assets remain under their upstream licenses.

## CD45 input

The included constraint is [config/targets/cd45.json](../config/targets/cd45.json).
Its target label is `cd45`, and its recorded target ID is `05_CD45`.
After asset configuration, set `TARGET=cd45` in the campaign `.env`.
The generated `.env.assets` supplies `TREX_TARGET_ASSET_ROOT`; the launcher
selects `bindcraft_targets/CD45.pdb` under that root and the included target
constraints automatically. Explicit `TREX_TARGET_PDB` or `TREX_TARGET_CONFIG`
values override this lookup; use them only when intentionally selecting a
different input, as described in the [custom-target guide](../examples/custom_target/README.md).

Use the CD45 structure from the study's target-data tree, with the original
chains and residue numbering. Its expected size is 115,829 bytes and SHA256 is:

```text
ea9e068e7f44d29d51e05e639bb361015350e26b4efdf45a67f8fc62c624efaa
```

The registry and hashes are in
[registry.json](../config/targets/registry.json) and
[assets.sha256.json](../config/targets/assets.sha256.json).
The exact CD45 bytes are available through the [asset collection](assets.md),
under `targets/bindcraft_targets/CD45.pdb`. The downloader uses a public
Complexa revision verified to contain the same bytes.
A different PDB crop is a different input, even if it has the same target name.

## Local LLM and clustering tools

The environment installed above contains vLLM 0.19.0 and PyTorch 2.10.0.

Download the full
[Qwen3.6-27B-FP8 snapshot](https://huggingface.co/Qwen/Qwen3.6-27B-FP8/tree/ec4160bf26124fa57e6451d070ee0c459a36d5b7)
at revision `ec4160bf26124fa57e6451d070ee0c459a36d5b7`, using the Hugging Face
CLI from the activated serving environment (skip this download if the asset
manager has already provided the model):

```bash
source .venv-serving/bin/activate
hf download Qwen/Qwen3.6-27B-FP8 \
  --revision ec4160bf26124fa57e6451d070ee0c459a36d5b7 \
  --local-dir /absolute/path/to/Qwen3.6-27B-FP8
```

Set `TREX_QWEN_MODEL_PATH` to that directory and keep the included
[model manifest](../config/trex/qwen3_6_27b_fp8_model_manifest.json).
The Slurm script starts the server automatically with context length 65,536,
GPU-memory utilization 0.90 and thinking disabled.
The first startup compiles GPU kernels and can take several minutes; subsequent
starts reuse the compiled cache. Startup logs are saved in the campaign archive.

Install [Foldseek](https://github.com/steineggerlab/foldseek) and
[MMseqs2](https://github.com/soedinglab/MMseqs2), then set their absolute
executable paths in `TREX_FOLDSEEK_BIN` and `TREX_MMSEQS_BIN`.
Recorded versions are `8dc75c74ad0eddab73cfd905963d13bf74dc012b` and
`76da68ad7577378410c075049e18666fcc94f8d1`, respectively.
Foldseek is required for structural uniqueness; MMseqs2 provides sequence
diversity and is also used by the panel command.

Continue with [CD45 configuration, preflight and Slurm submission](slurm.md).
