# Checkpoints and target structures

The main [README](../README.md#1-install-t-rex-backends-and-checkpoints)
downloads, verifies and configures this collection as part of `trex setup`.
The commands below are component-level alternatives for mirrors, partial
downloads and audits.

Use the asset manager from a T-REX source checkout with Python 3.10–3.12.
Upstream downloads use the Python standard library; Google Drive downloads
use the optional `assets` dependency. The pinned manifest records each
file's byte size, SHA256, source URL and component license.

```bash
python scripts/manage_assets.py list
python scripts/manage_assets.py fetch --root ../T-REX-assets
python scripts/manage_assets.py verify --root ../T-REX-assets
```

The full collection is about **52.0 GB (48.4 GiB)**. The command above uses
pinned upstream URLs. The public ZIP below is the simplest full-bundle download.
Both paths validate bytes before committing files and reuse existing files only
after checking their SHA256. A conflicting or corrupt existing file causes an
error; move it aside before retrying. No model code is loaded or executed.

For a smaller download, repeat `--component` as needed:

```bash
python scripts/manage_assets.py fetch --root ../T-REX-assets \
  --component targets --component proteinmpnn
```

| Component | Approximate size | Contents |
| --- | ---: | --- |
| `complexa` | 7.03 GB | Protein-target checkpoint and autoencoder |
| `af2` | 5.59 GB | 15 AF2 parameter files, shared by AF2 evaluation, Complexa and BindCraft |
| `proteinmpnn` | 0.03 GB | Four vanilla ProteinMPNN weight files |
| `colabdesign_bindcraft` | 0.05 GB | Eight converted ProteinMPNN weights, AF2 template array and licenses, matching BindCraft's installed package |
| `documentation` | <0.001 GB | Setup, version records, citations, notices, compatible environment locks and required patches |
| `boltzgen` | 8.41 GB | Five cached checkpoints, molecule dictionary and fixed HF cache references |
| `qwen` | 30.89 GB | Exact Qwen3.6-27B-FP8 model snapshot |
| `targets` | 0.002 GB | Seven study structures, preserving the registered crops, chains and residue numbering |

The target bundle contains BetV1, CbAgo, CD45, HER2, SC2RBD, IL7RA and PD-L1. The `her2aav` target label refers to the human HER2
structure. These files cover all seven primary benchmark targets.

The BoltzGen collection includes the model cache and the
optional affinity checkpoint; not every workflow uses every cached model.
Unused Complexa ligand/AME checkpoints and unrelated model caches are excluded.
Additional models used only in separate post-hoc analyses are outside this
production-controller asset collection.

## Download the ZIP from Google Drive

A public ZIP is available from the file-specific Google Drive link below.
The release size and SHA256 are pinned in
[assets_release.json](../config/reproducibility/assets_release.json).

```bash
python -m pip install -e '.[assets]'
TREX_ASSET_DRIVE_URL='https://drive.google.com/file/d/1QkXn7AoHD-pasiHoIx5o08TlTrWjrqKx/view?usp=share_link'
python scripts/manage_assets.py fetch --root ../T-REX-assets \
  --drive-url "$TREX_ASSET_DRIVE_URL"
```

The downloader resumes supported interrupted transfers, checks the entire ZIP
against the release hash, then checks each extracted file against the asset
manifest. It keeps the ZIP in `../.trex-downloads/` for reuse. Allow at least
**110 GB of free space** for the ZIP and extracted assets, plus space for the
software environments and campaign outputs. Google Drive access restrictions
or download quotas can stop a transfer; direct upstream downloads remain
available with the first command on this page.

If you downloaded the ZIP manually, import it with the same per-file checks:

```bash
python scripts/manage_assets.py import-zip \
  --archive /absolute/path/to/T-REX-assets.zip --root ../T-REX-assets
```

Imports install manifest-declared files, including setup/version/citation documents,
and reject path traversal, symlinks, duplicate ZIP entries and conflicting existing
files. ZIP64 support is required for this large archive.

## Connect assets to backend checkouts

First obtain the backend source trees described in the
[installation guide](installation.md#backend-source-and-environments).
The public Complexa checkout also supplies the study's AF2/ProteinMPNN source.

For existing source trees, run:

```bash
python scripts/manage_assets.py configure \
  --root ../T-REX-assets \
  --complexa-repo ./external/Proteina-Complexa \
  --community-repo ./external/Proteina-Complexa \
  --bindcraft-repo ./external/BindCraft \
  --env-out .env.assets
source .env.assets
```

`configure` rehashes the checkpoints it links, checks for conflicts, then connects
the weights using symlinks. AF2 files are shared by Complexa and BindCraft.
Existing byte-identical files are kept; mismatched files and settings
are never overwritten. Repeating the command with the same arguments is safe.
Keep the asset folder in place after configuring it.

BindCraft's ColabDesign package supplies its own `.pkl` weights. The bundle
retains exact copies; `configure` does not modify Python packages. After installing
the pinned package in the [installation guide](installation.md#check-each-backend-environment),
verify the installed weights with:

```bash
python scripts/manage_assets.py verify-bindcraft \
  --python /absolute/path/to/BindCraft/environment/bin/python
```

The generated `.env.assets` exports asset and source-tree paths only. It does
not install Python environments, PyRosetta, Foldseek, MMseqs2 or GPU drivers,
and does not start the LLM server. Configure executable paths as described in
[installation](installation.md), then run the documented preflight. If using a
campaign YAML, set its explicit backend/model paths to this same installation;
explicit YAML values take precedence over environment defaults.

## Local bundles and a publication mirror

A collected bundle uses this layout:

```text
T-REX-assets/
  manifest.json
  README.md
  THIRD_PARTY_NOTICES.md
  SETUP.md
  VERSIONS.json
  CITATIONS.md
  setup/
  checkpoints/
    complexa/
    af2/
    proteinmpnn/
    colabdesign_bindcraft/
    boltzgen/
    Qwen3.6-27B-FP8/
  targets/
    bindcraft_targets/
    alpha_proteo_targets/
```

To copy a downloaded bundle into another location with content verification:

```bash
python scripts/manage_assets.py fetch --root ../my-assets \
  --source-dir /absolute/path/to/T-REX-assets
```

`fetch --base-url HTTPS_URL` supports a mirror preserving the manifest paths.
Use a release-specific URL or a fixed Hugging Face dataset commit. The mirror
location has not been published yet; the default upstream URLs can already be
used. Publish the README, manifest, notices and component license files with the
weights. The collection has multiple upstream licenses; T-REX's MIT license
does not replace them.

The release size and hash are recorded in `assets_release.json`. Upload the ZIP
unchanged so that hash remains valid. The extracted collection is about 52.01 GB.
See [upstream citations](citations.md) for official repositories and papers.

Use matching code and asset releases: `SETUP.md`, `VERSIONS.json` and the
`setup/` installation profiles are included in the manifest. An older bundle
with different file contents fails the current manifest check.
For complete installation commands, follow the [installation guide](installation.md).

## Verification scope

Release checks cover ZIP integrity and each declared file's SHA256. File
integrity does not establish software installation or inference success; see
[installation](installation.md) and [reproducibility](reproducibility.md).
Google Drive download validation must be repeated after replacing the file at
the public link with this release. The default pinned upstream download route
is independent of Drive.
