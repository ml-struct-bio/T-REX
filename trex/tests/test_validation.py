from __future__ import annotations

import hashlib
import json
from pathlib import Path

from trex import validation


_ONE_RESIDUE_PDB = """\
ATOM      1  N   ALA A   1      11.104  13.207   9.121  1.00 20.00           N
ATOM      2  CA  ALA A   1      12.560  13.207   9.121  1.00 20.00           C
ATOM      3  C   ALA A   1      13.000  14.620   9.500  1.00 20.00           C
TER
END
"""


def test_custom_target_works_without_installed_registry(
    tmp_path: Path, monkeypatch
) -> None:
    config = tmp_path / "custom.json"
    pdb = tmp_path / "custom.pdb"
    config.write_text(
        json.dumps(
            {
                "target_id": "custom_001",
                "target_class": "protein",
                "chain_ids": ["A"],
                "hotspots": ["A1"],
                "panel_size": 8,
            }
        )
    )
    pdb.write_text(_ONE_RESIDUE_PDB)
    monkeypatch.setattr(validation, "TARGET_REGISTRY", tmp_path / "missing.json")

    checks = validation.validate_install(
        target="custom",
        enabled_families=(),
        target_config=config,
        target_pdb=pdb,
    )

    assert not [check for check in checks if check.required and not check.passed]
    assert any(
        check.name == "target registry" and check.status == "warn" for check in checks
    )


def test_registered_target_without_registry_fails_cleanly(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(validation, "TARGET_REGISTRY", tmp_path / "missing.json")

    checks = validation.validate_install(target="cd45", enabled_families=())

    assert checks[0].name == "target registry"
    assert checks[0].status == "fail"
    assert "source checkout" in checks[0].detail


def test_require_model_is_fatal_without_model_path(monkeypatch) -> None:
    monkeypatch.delenv("TREX_QWEN_MODEL_PATH", raising=False)

    checks = validation.validate_install(
        target="cd45",
        enabled_families=(),
        require_model=True,
        verify_asset_hash=False,
    )

    failures = [check for check in checks if check.required and check.status == "fail"]
    assert any(check.name == "Qwen model directory" for check in failures)


def test_backend_revision_check_verifies_archived_patch(
    tmp_path: Path, monkeypatch
) -> None:
    config = tmp_path / "custom.json"
    pdb = tmp_path / "custom.pdb"
    config.write_text(
        json.dumps(
            {
                "target_id": "custom_001",
                "target_class": "protein",
                "chain_ids": ["A"],
                "hotspots": ["A1"],
                "panel_size": 8,
            }
        )
    )
    pdb.write_text(_ONE_RESIDUE_PDB)
    patch = tmp_path / "bindcraft.patch"
    patch.write_text("mode-only production patch\n")
    patch_sha = hashlib.sha256(patch.read_bytes()).hexdigest()
    stack = tmp_path / "production_stack.json"
    stack.write_text(
        json.dumps(
            {
                "components": {
                    "bindcraft": {
                        "git_commit": "expected-head",
                        "patch": patch.name,
                        "patch_file_sha256": patch_sha,
                        "production_tracked_diff_sha256": "expected-diff",
                    }
                }
            }
        )
    )
    bindcraft = tmp_path / "BindCraft"
    bindcraft.mkdir()
    monkeypatch.setenv("TREX_BINDCRAFT_REPO", str(bindcraft))
    monkeypatch.setattr(validation, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(validation, "PRODUCTION_STACK", stack)
    monkeypatch.setattr(validation, "_git_head", lambda _path: "expected-head")
    monkeypatch.setattr(validation, "_git_diff_sha256", lambda _path: "expected-diff")

    checks = validation.validate_install(
        target="custom",
        target_config=config,
        target_pdb=pdb,
        enabled_families=("bindcraft",),
        verify_backend_revisions=True,
    )
    patch_check = next(
        check for check in checks if check.name == "bindcraft patch artifact"
    )
    assert patch_check.status == "ok"

    patch.write_text("changed\n")
    checks = validation.validate_install(
        target="custom",
        target_config=config,
        target_pdb=pdb,
        enabled_families=("bindcraft",),
        verify_backend_revisions=True,
    )
    patch_check = next(
        check for check in checks if check.name == "bindcraft patch artifact"
    )
    assert patch_check.status == "fail"


def test_validation_json_output_is_versioned(monkeypatch, capsys) -> None:
    monkeypatch.setattr(validation, "validate_install", lambda **_kwargs: [])

    assert validation.main(["--target", "test", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "trex.installation-validation.v1"
    assert payload["failure_count"] == 0
    assert payload["checks"] == []
