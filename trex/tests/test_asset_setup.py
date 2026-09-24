"""Asset transport tests use synthetic bytes, never model execution."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import pytest

SPEC = importlib.util.spec_from_file_location(
    "manage_assets", Path(__file__).resolve().parents[2] / "scripts/manage_assets.py"
)
assets = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(assets)


def entry(path, data=b"synthetic asset", **kwargs):
    return {
        "path": path,
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        **kwargs,
    }


def manifest(*entries):
    return {
        "schema_version": "trex.assets.v1",
        "components": {"fixture": {"files": list(entries)}},
    }


@pytest.mark.parametrize(
    "name",
    [
        "../escape",
        "/absolute",
        "a/../../b",
        "a//b",
        "a/./b",
        "a\\b",
        "",
        ".",
        "x\ny",
        "C:/file",
    ],
)
def test_reject_unsafe_manifest_paths(tmp_path, name):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest(entry(name))))
    with pytest.raises(ValueError):
        assets.load_manifest(path)


def test_duplicate_paths_rejected(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest(entry("same"), entry("same"))))
    with pytest.raises(ValueError, match="Duplicate"):
        assets.load_manifest(path)


def test_local_fetch_idempotent_and_hashes_content(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model").write_bytes(b"synthetic asset")
    root = tmp_path / "dest"
    components = manifest(entry("model"))["components"]
    assets.fetch(root, components, base_url=None, source_dir=source)
    original_inode = (root / "model").stat().st_ino
    assets.fetch(root, components, base_url=None, source_dir=source)
    assert (root / "model").stat().st_ino == original_inode
    assert (root / "model").read_bytes() == (source / "model").read_bytes()
    (root / "model").write_bytes(b"synthetic wrong")
    with pytest.raises(ValueError, match="SHA256"):
        assets.fetch(root, components, base_url=None, source_dir=source)
    assert (root / "model").read_bytes() == b"synthetic wrong"


@pytest.mark.parametrize(
    "wrong", [b"short", b"much longer than the expected size", b"synthetic wrong"]
)
def test_bad_download_never_committed(tmp_path, wrong):
    target = tmp_path / "model"
    with pytest.raises(ValueError):
        assets.install_stream(io.BytesIO(wrong), target, entry("model"))
    assert not target.exists()
    assert not list(tmp_path.glob(".trex-asset-*"))


def test_interrupted_transfer_cleans_temporary(tmp_path):
    class Interrupted:
        def read(self, size):
            raise OSError("connection interrupted")

    with pytest.raises(OSError):
        assets.install_stream(Interrupted(), tmp_path / "model", entry("model"))
    assert not list(tmp_path.iterdir())


def test_symlink_escape_is_rejected(tmp_path):
    root = tmp_path / "assets"
    root.mkdir()
    (root / "outside").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        assets.asset_path(root, "outside/model")


def test_concurrent_destination_is_preserved(tmp_path):
    dest = tmp_path / "model"
    dest.write_bytes(b"other process")
    with pytest.raises(FileExistsError):
        assets.install_stream(io.BytesIO(b"synthetic asset"), dest, entry("model"))
    assert dest.read_bytes() == b"other process"


def archive_bytes(*members):
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w") as archive:
        for name, data, link in members:
            info = tarfile.TarInfo(name)
            if link:
                info.type = tarfile.SYMTYPE
                info.linkname = "../../outside"
                archive.addfile(info)
            else:
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
    return out.getvalue()


def test_archive_extracts_only_declared_regular_members(tmp_path, monkeypatch):
    content = archive_bytes(
        ("../../outside", b"ignored", False), ("./model", b"synthetic asset", False)
    )
    monkeypatch.setattr(assets, "open_url", lambda _: io.BytesIO(content))
    assets.fetch_archive(
        tmp_path,
        "https://example.test/archive",
        [entry("weights/model", archive_member="model")],
    )
    assert (tmp_path / "weights/model").read_bytes() == b"synthetic asset"
    assert sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*")) == [
        "weights",
        "weights/model",
    ]


def test_archive_rejects_declared_symlink(tmp_path, monkeypatch):
    content = archive_bytes(("model", b"", True))
    monkeypatch.setattr(assets, "open_url", lambda _: io.BytesIO(content))
    with pytest.raises(ValueError, match="Invalid archive"):
        assets.fetch_archive(
            tmp_path,
            "https://example.test/archive",
            [entry("model", archive_member="model")],
        )
    assert not (tmp_path / "model").exists()


def test_archive_missing_member_fails(tmp_path, monkeypatch):
    content = archive_bytes(("other", b"ignored", False))
    monkeypatch.setattr(assets, "open_url", lambda _: io.BytesIO(content))
    with pytest.raises(ValueError, match="missing"):
        assets.fetch_archive(
            tmp_path,
            "https://example.test/archive",
            [entry("model", archive_member="model")],
        )


def setup_fixture(tmp_path):
    root = tmp_path / "asset folder"
    paths = {
        "complexa": "checkpoints/complexa/complexa.ckpt",
        "af2": "checkpoints/af2/params_model_1.npz",
        "proteinmpnn": "checkpoints/proteinmpnn/v_48_020.pt",
        "boltzgen": "checkpoints/boltzgen/fixture",
        "qwen": "checkpoints/Qwen3.6-27B-FP8/fixture",
        "targets": "targets/toy.txt",
    }
    m = {"components": {}}
    for key, path in paths.items():
        f = root / path
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"synthetic asset")
        m["components"][key] = {"files": [entry(path)]}
    backends = [tmp_path / name for name in ("complexa", "community", "bindcraft")]
    for backend in backends:
        backend.mkdir()
    return root, m, backends, tmp_path / "env file.sh"


@pytest.mark.parametrize("shared_checkout", [False, True])
def test_configure_shared_af2_and_quoted_paths_idempotent(tmp_path, shared_checkout):
    root, m, backends, env = setup_fixture(tmp_path)
    if shared_checkout:
        backends[1] = backends[0]
    assets.configure(root, m, *backends, env)
    assert (
        backends[0] / "community_models/ckpts/AF2/params_model_1.npz"
    ).resolve() == (backends[2] / "params/params_model_1.npz").resolve()
    assert "export TREX_TARGET_ASSET_ROOT='" in env.read_text()
    original = env.read_bytes()
    assets.configure(root, m, *backends, env)
    assert env.read_bytes() == original


def test_configure_checks_all_conflicts_before_writing(tmp_path):
    root, m, backends, env = setup_fixture(tmp_path)
    collision = backends[2] / "params/params_model_1.npz"
    collision.parent.mkdir()
    collision.write_bytes(b"old data")
    with pytest.raises(ValueError, match="mismatch"):
        assets.configure(root, m, *backends, env)
    assert not (backends[0] / "ckpts").exists()
    assert not env.exists()
    assert collision.read_bytes() == b"old data"


def test_configure_environment_conflict_preserves_existing(tmp_path):
    root, m, backends, env = setup_fixture(tmp_path)
    env.write_text("existing settings\n")
    with pytest.raises(ValueError, match="overwrite"):
        assets.configure(root, m, *backends, env)
    assert not (backends[0] / "ckpts").exists()
    assert env.read_text() == "existing settings\n"


def test_configure_env_cannot_collide_with_planned_link(tmp_path):
    root, m, backends, _ = setup_fixture(tmp_path)
    with pytest.raises(ValueError, match="conflicts"):
        assets.configure(root, m, *backends, backends[0] / "ckpts/complexa.ckpt")
    assert not (backends[0] / "ckpts").exists()


def test_real_manifest_is_portable_and_has_expected_components():
    m = assets.load_manifest(assets.DEFAULT_MANIFEST)
    assert set(m["components"]) == {
        "complexa",
        "af2",
        "proteinmpnn",
        "boltzgen",
        "qwen",
        "targets",
        "colabdesign_bindcraft",
        "documentation",
    }
    text = assets.DEFAULT_MANIFEST.read_text()
    assert "/scratch/" not in text and "/home/" not in text
    target_files = {
        f["target"]: f
        for f in m["components"]["targets"]["files"]
        if f["path"].endswith(".pdb")
    }
    assert set(target_files) == {
        "betv1",
        "cbago",
        "cd45",
        "her2aav",
        "il7ra",
        "pdl1",
        "sc2rbd",
    }
    root = Path(__file__).resolve().parents[2]
    registry = json.loads((root / "config/targets/registry.json").read_text())[
        "targets"
    ]
    hashes = json.loads((root / "config/targets/assets.sha256.json").read_text())[
        "targets"
    ]
    assert set(registry) == set(hashes) == set(target_files) | {"sc2rbd"}
    for name, entry in target_files.items():
        assert entry["path"] == "targets/" + registry[name]["pdb"]
        assert entry["sha256"] == hashes[name]["sha256"]
        assert entry["size"] == hashes[name]["bytes"]
    for prefix in ("config/targets", "trex/data/config/targets"):
        configs = set((root / prefix).glob("*.json"))
        assert {p.name for p in configs} == {
            "registry.json",
            "assets.sha256.json",
            *(entry["config"] for entry in registry.values()),
        }
    for entry in registry.values():
        relative = Path("config/targets") / entry["config"]
        assert (root / relative).read_bytes() == (
            root / "trex/data" / relative
        ).read_bytes()


def installed_colabdesign_fixture(tmp_path):
    relative = "colabdesign/mpnn/weights_soluble/v_48_020.pkl"
    # Deliberately not a serialized model: verification must only hash bytes.
    data = b"not a pickle; verification must not execute it"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(data)
    component = {
        "package_version": "1.1.3",
        "files": [entry("checkpoints/colabdesign_bindcraft/" + relative, data)],
    }
    return {"components": {"colabdesign_bindcraft": component}}, path


def mock_package_metadata(monkeypatch, tmp_path, version="1.1.3"):
    import types

    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert "-I" in command
        assert "import colabdesign" not in command[-1]
        return types.SimpleNamespace(
            stdout=json.dumps({"version": version, "site": str(tmp_path)})
        )

    monkeypatch.setattr(assets.subprocess, "run", run)
    return calls


def test_verify_installed_colabdesign_weights_without_deserialization(
    tmp_path, monkeypatch
):
    m, _ = installed_colabdesign_fixture(tmp_path)
    calls = mock_package_metadata(monkeypatch, tmp_path)
    python = tmp_path / "environment/bin/python"
    assets.verify_bindcraft(m, python)
    assert calls[0][0] == str(python)


@pytest.mark.parametrize("failure", ["missing", "corrupt", "version"])
def test_verify_installed_colabdesign_rejects_mismatch(tmp_path, monkeypatch, failure):
    m, path = installed_colabdesign_fixture(tmp_path)
    mock_package_metadata(
        monkeypatch, tmp_path, "different" if failure == "version" else "1.1.3"
    )
    if failure == "missing":
        path.unlink()
    elif failure == "corrupt":
        path.write_bytes(b"x" * path.stat().st_size)
    with pytest.raises(ValueError):
        assets.verify_bindcraft(m, tmp_path / "python")


def test_documentation_fetch_and_import_preserve_notices_and_setup(tmp_path):
    import zipfile

    m = assets.load_manifest(assets.DEFAULT_MANIFEST)
    components = assets.selected_components(m, ["documentation"])
    fetched = tmp_path / "fetched"
    assets.fetch(fetched, components, base_url=None, source_dir=None)
    archive = tmp_path / "docs.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for entry_ in components["documentation"]["files"]:
            z.write(fetched / entry_["path"], "T-REX-assets/" + entry_["path"])
    installed = tmp_path / "installed"
    assets.import_zip(installed, archive, components)
    for name in [
        "README.md",
        "SETUP.md",
        "VERSIONS.json",
        "CITATIONS.md",
        "THIRD_PARTY_NOTICES.md",
    ]:
        assert (installed / name).read_bytes() == (fetched / name).read_bytes()
    versions = json.loads((installed / "VERSIONS.json").read_text())
    lock = installed / versions["default_serving_profile"]["lock"]
    assert (
        hashlib.sha256(lock.read_bytes()).hexdigest()
        == versions["default_serving_profile"]["lock_sha256"]
    )


@pytest.mark.parametrize(
    "bundle_path,source_path",
    [
        ("setup/serving_requirements.lock.txt", "config/trex/serving_requirements.lock.txt"),
        ("setup/complexa_requirements.lock.txt", "config/trex/complexa_requirements.lock.txt"),
        ("setup/bindcraft_requirements.lock.txt", "config/trex/bindcraft_requirements.lock.txt"),
        ("setup/boltzgen_requirements.lock.txt", "config/trex/boltzgen_requirements.lock.txt"),
        ("setup/complexa-dependencies.patch", "external/patches/complexa-dependencies.patch"),
        ("setup/bindcraft-production.patch", "external/patches/bindcraft-production.patch"),
    ],
)
def test_bundled_installation_inputs_match_checkout(bundle_path, source_path):
    manifest = assets.load_manifest(assets.DEFAULT_MANIFEST)
    entries = {e["path"]: e for e in manifest["components"]["documentation"]["files"]}
    entry = entries[bundle_path]
    content = (assets.REPO / source_path).read_bytes()
    assert entry["text"].encode() == content
    assert entry["size"] == len(content)
    assert entry["sha256"] == hashlib.sha256(content).hexdigest()


def test_bundled_profiles_and_validation_scope_are_current():
    manifest = assets.load_manifest(assets.DEFAULT_MANIFEST)
    docs = {e["path"]: e for e in manifest["components"]["documentation"]["files"]}
    versions = json.loads(docs["VERSIONS.json"]["text"])
    for name in ("serving", "complexa"):
        profile = json.loads(docs[f"setup/{name}_environment.json"]["text"])
        expected = json.loads(
            (assets.REPO / f"config/reproducibility/{name}_environment.json").read_text()
        )
        expected["lock"] = f"setup/{name}_requirements.lock.txt"
        if name == "serving":
            expected["historical_inventory"] = (
                "Study inventory is recorded in the code release; use this profile for installation."
            )
        else:
            expected["dependency_patch"] = "setup/complexa-dependencies.patch"
        assert profile == expected == versions[f"default_{name}_profile"]
        assert profile["lock_sha256"] == docs[profile["lock"]]["sha256"]
    backends = versions["observed_backend_environments"]
    for backend in backends.values():
        assert backend["fresh_install_validated"] is True
        assert backend["dependency_check"] == "passed: zero conflicts"
        assert backend["lock_sha256"] == docs[backend["lock"]]["sha256"]
    assert backends["bindcraft"]["whole_job_completed"] is False
    assert backends["boltzgen"]["whole_job_completed"] is True
    assert versions["validation_scope"]["immutable_final_release_replay"] is False
    assert "T-REX-publication/scripts" not in docs["SETUP.md"]["text"]
    assert "11 metadata" not in docs["SETUP.md"]["text"]


def test_asset_release_file_counts_match_manifest():
    manifest = assets.load_manifest(assets.DEFAULT_MANIFEST)
    release = json.loads(
        (assets.REPO / "config/reproducibility/assets_release.json").read_text()
    )
    count = sum(len(component["files"]) for component in manifest["components"].values())
    assert release["release"] == manifest["release"]
    assert release["manifest_sha256"] == hashlib.sha256(
        assets.DEFAULT_MANIFEST.read_bytes()
    ).hexdigest()
    assert release["manifest_files"] == count
    # The manifest, checksum inventory and Git LFS attributes are ZIP metadata.
    assert release["total_archive_files"] == count + 3
