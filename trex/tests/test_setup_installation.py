from __future__ import annotations

from pathlib import Path

import pytest

from trex.setup_installation import (
    MANAGED_HEADER,
    SetupError,
    _assets_complete,
    _complexa_environment,
    _managed_write,
)


def test_assets_complete_checks_every_manifest_file(tmp_path: Path) -> None:
    root = tmp_path / "assets"
    root.mkdir()
    item = root / "checkpoint.bin"
    item.write_bytes(b"1234")
    manifest = {
        "components": {
            "demo": {
                "files": [{"path": "checkpoint.bin", "size": 4, "sha256": "unused"}]
            }
        }
    }

    assert _assets_complete(root, manifest)
    item.write_bytes(b"123")
    assert not _assets_complete(root, manifest)


def test_managed_write_preserves_user_configuration(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("USER_SETTING=1\n")

    with pytest.raises(SetupError, match="refusing to overwrite"):
        _managed_write(path, MANAGED_HEADER + "GENERATED=1\n")

    assert path.read_text() == "USER_SETTING=1\n"


def test_complexa_environment_uses_absolute_installed_paths(tmp_path: Path) -> None:
    text = _complexa_environment(
        tmp_path / "Complexa",
        tmp_path / "assets",
        tmp_path / "bin/foldseek",
        tmp_path / "bin/mmseqs",
        tmp_path / "BindCraft",
    )

    assert text.startswith(MANAGED_HEADER)
    assert f"LOCAL_CODE_PATH={tmp_path / 'Complexa'}" in text
    assert f"UV_FOLDSEEK_EXEC={tmp_path / 'bin/foldseek'}" in text
    assert f"UV_MMSEQS_EXEC={tmp_path / 'bin/mmseqs'}" in text
    assert f"UV_DSSP_EXEC={tmp_path / 'BindCraft/functions/dssp'}" in text
