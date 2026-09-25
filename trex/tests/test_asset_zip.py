"""ZIP and Drive transport regressions, using only synthetic file bytes."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import stat
import sys
import types
import zipfile

import pytest

spec = importlib.util.spec_from_file_location(
    "asset_zip_tool", Path(__file__).resolve().parents[2] / "scripts/manage_assets.py"
)
assets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assets)
DATA = b"synthetic weights"
ENTRY = {
    "path": "checkpoints/toy",
    "size": len(DATA),
    "sha256": hashlib.sha256(DATA).hexdigest(),
}
COMPONENTS = {"toy": {"files": [ENTRY]}}


def make_zip(tmp_path, name="T-REX-assets/checkpoints/toy", data=DATA):
    path = tmp_path / "release.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(name, data)
    return path


def test_zip_round_trip_and_idempotence(tmp_path):
    path = make_zip(tmp_path)
    root = tmp_path / "out"
    assets.import_zip(root, path, COMPONENTS)
    output = root / ENTRY["path"]
    inode = output.stat().st_ino
    assets.import_zip(root, path, COMPONENTS)
    assert output.read_bytes() == DATA
    assert output.stat().st_ino == inode


@pytest.mark.parametrize(
    "name",
    [
        "../outside",
        "/absolute",
        "T-REX-assets/../../outside",
        "other/checkpoints/toy",
        "T-REX-assets/a\\b",
    ],
)
def test_zip_paths_rejected_before_writing(tmp_path, name):
    path = make_zip(tmp_path, name)
    root = tmp_path / "out"
    with pytest.raises(ValueError):
        assets.import_zip(root, path, COMPONENTS)
    assert not root.exists()


def test_zip_symlink_rejected(tmp_path):
    path = tmp_path / "release.zip"
    with zipfile.ZipFile(path, "w") as z:
        info = zipfile.ZipInfo("T-REX-assets/checkpoints/toy")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        z.writestr(info, "../../outside")
    with pytest.raises(ValueError, match="symlink"):
        assets.import_zip(tmp_path / "out", path, COMPONENTS)


def test_zip_duplicate_member_rejected(tmp_path):
    path = make_zip(tmp_path)
    with zipfile.ZipFile(path, "a") as z, pytest.warns(UserWarning):
        z.writestr("T-REX-assets/checkpoints/toy", DATA)
    with pytest.raises(ValueError, match="Duplicate"):
        assets.import_zip(tmp_path / "out", path, COMPONENTS)


def test_zip_wrong_bytes_not_committed(tmp_path):
    path = make_zip(tmp_path, data=b"synthetic invalid")
    with pytest.raises(ValueError, match="match manifest"):
        assets.import_zip(tmp_path / "out", path, COMPONENTS)
    assert not (tmp_path / "out/checkpoints/toy").exists()


def test_zip_keeps_existing_conflicting_file(tmp_path):
    path = make_zip(tmp_path)
    root = tmp_path / "out"
    output = root / ENTRY["path"]
    output.parent.mkdir(parents=True)
    output.write_bytes(b"existing different bytes")
    with pytest.raises(ValueError, match="mismatch"):
        assets.import_zip(root, path, COMPONENTS)
    assert output.read_bytes() == b"existing different bytes"


def drive_fixture(tmp_path):
    archive = make_zip(tmp_path)
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    release = tmp_path / "release.json"
    release.write_text(json.dumps({"sha256": sha, "size": archive.stat().st_size}))
    return archive, release, sha


def test_drive_checks_archive_then_members_and_reuses_assets(tmp_path, monkeypatch):
    archive, release, sha = drive_fixture(tmp_path)
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        shutil.copyfile(archive, kwargs["output"])
        return kwargs["output"]

    monkeypatch.setitem(sys.modules, "gdown", types.SimpleNamespace(download=download))
    root = tmp_path / "out"
    for _ in range(2):
        assets.fetch_drive(
            root,
            COMPONENTS,
            "https://drive.google.com/file/d/example/view",
            release,
            tmp_path / "cache",
        )
    assert len(calls) == 1
    assert calls[0] == {
        "url": "https://drive.google.com/file/d/example/view",
        "output": str(tmp_path / "cache" / (sha + ".zip")),
        "resume": True,
        "use_cookies": False,
        "fuzzy": True,
    }
    assert (root / ENTRY["path"]).read_bytes() == DATA


def test_drive_replaces_bad_cached_archive_before_extraction(tmp_path, monkeypatch):
    archive, release, sha = drive_fixture(tmp_path)
    cache = tmp_path / "cache"
    cache.mkdir()
    cached = cache / (sha + ".zip")
    cached.write_bytes(b"x" * archive.stat().st_size)
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        assert not cached.exists()
        shutil.copyfile(archive, kwargs["output"])
        return kwargs["output"]

    monkeypatch.setitem(sys.modules, "gdown", types.SimpleNamespace(download=download))
    root = tmp_path / "out"
    assets.fetch_drive(
        root,
        COMPONENTS,
        "https://drive.google.com/file/d/example/view",
        release,
        cache,
    )
    assert len(calls) == 1
    assert (root / ENTRY["path"]).read_bytes() == DATA


def test_drive_rejects_bad_download_before_extraction(tmp_path, monkeypatch):
    archive, release, _ = drive_fixture(tmp_path)

    def download(**kwargs):
        Path(kwargs["output"]).write_bytes(b"x" * archive.stat().st_size)
        return kwargs["output"]

    monkeypatch.setitem(sys.modules, "gdown", types.SimpleNamespace(download=download))
    with pytest.raises(ValueError, match="SHA256"):
        assets.fetch_drive(
            tmp_path / "out",
            COMPONENTS,
            "https://drive.google.com/file/d/example/view",
            release,
            tmp_path / "cache",
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "url",
    [
        "http://drive.google.com/file/d/test",
        "https://example.com/test",
        "https://drive.google.com/drive/folders/test",
    ],
)
def test_drive_rejects_non_file_source(tmp_path, url):
    with pytest.raises(ValueError):
        assets.fetch_drive(
            tmp_path / "out", COMPONENTS, url, tmp_path / "absent", tmp_path / "cache"
        )


def test_drive_reports_transport_failure_without_extracting(tmp_path, monkeypatch):
    _, release, _ = drive_fixture(tmp_path)

    def fail(**kwargs):
        raise RuntimeError("quota exceeded")

    monkeypatch.setitem(sys.modules, "gdown", types.SimpleNamespace(download=fail))
    with pytest.raises(
        ValueError, match="Drive download failed \\(RuntimeError: quota exceeded\\)"
    ):
        assets.fetch_drive(
            tmp_path / "out",
            COMPONENTS,
            "https://drive.google.com/file/d/test/view",
            release,
            tmp_path / "cache",
        )
    assert not (tmp_path / "out").exists()
