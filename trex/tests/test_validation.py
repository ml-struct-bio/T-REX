from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

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


@pytest.mark.parametrize(
    "variant",
    ["equivalent", "historical", "dirty", "changed_tree", "unlisted", "no_trees"],
)
def test_community_revision_requires_equivalent_source(tmp_path, monkeypatch, variant):
    checkout = tmp_path / "community"
    checkout.mkdir()

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=checkout, check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init", "-q")
    source = checkout / "community_models/colabdesign"
    source.mkdir(parents=True)
    (source / "__init__.py").write_text("version = 1\n")
    git("add", ".")
    git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.org",
        "commit",
        "-qm",
        "source",
    )
    historical = git("rev-parse", "HEAD")
    tree = git("rev-parse", "HEAD:community_models/colabdesign")
    (checkout / "README.md").write_text("Public checkout\n")
    git("add", ".")
    git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.org",
        "commit",
        "-qm",
        "docs",
    )
    public = git("rev-parse", "HEAD")
    if variant == "historical":
        git("checkout", "-q", historical)
    elif variant == "dirty":
        (source / "__init__.py").write_text("version = 2\n")
    component = {
        "git_commit": historical,
        "equivalent_git_commits": [] if variant == "unlisted" else [public],
        "equivalent_source_trees": {}
        if variant == "no_trees"
        else {
            "community_models/colabdesign": "0" * 40
            if variant == "changed_tree"
            else tree,
        },
    }
    stack = tmp_path / "stack.json"
    stack.write_text(
        json.dumps({"components": {"proteina_complexa_community": component}})
    )
    monkeypatch.setattr(validation, "PRODUCTION_STACK", stack)
    monkeypatch.setattr(
        validation, "_target_checks", lambda *a, **kw: ([], None, None, None)
    )
    monkeypatch.setenv("TREX_LEGACY_COMPLEXA_REPO", str(checkout))
    checks = validation.validate_install(
        target="test",
        enabled_families=("proteinmpnn_redesign",),
        verify_backend_revisions=True,
    )
    revision = next(
        check
        for check in checks
        if check.name == "proteina_complexa_community revision"
    )
    assert revision.passed == (variant in {"equivalent", "historical"})
    if variant == "equivalent":
        assert "identical declared source trees" in revision.detail


def test_mmseqs_is_required_for_a_full_backend_preflight(
    tmp_path: Path, monkeypatch
) -> None:
    foldseek = tmp_path / "foldseek"
    foldseek.write_text("#!/bin/sh\nexit 0\n")
    foldseek.chmod(0o755)
    monkeypatch.setattr(
        validation, "_target_checks", lambda *args, **kwargs: ([], None, None, None)
    )
    monkeypatch.setenv("TREX_FOLDSEEK_BIN", str(foldseek))
    monkeypatch.delenv("TREX_MMSEQS_BIN", raising=False)
    monkeypatch.setenv("PATH", "")

    checks = validation.validate_install(
        target="test", enabled_families=(), require_backends=True
    )

    mmseqs = next(check for check in checks if check.name == "MMseqs2 executable")
    assert mmseqs.required
    assert mmseqs.status == "fail"


def test_recorded_tool_versions_are_checked(tmp_path: Path, monkeypatch) -> None:
    foldseek_revision = "foldseek-recorded-revision"
    mmseqs_revision = "mmseqs-recorded-revision"
    executables = {}
    for name, revision in (
        ("foldseek", foldseek_revision),
        ("mmseqs", mmseqs_revision),
    ):
        executable = tmp_path / name
        executable.write_text(f"#!/bin/sh\necho {revision}\n")
        executable.chmod(0o755)
        executables[name] = executable
    stack = tmp_path / "stack.json"
    stack.write_text(
        json.dumps(
            {
                "components": {
                    "foldseek": {"version": foldseek_revision},
                    "mmseqs2": {"version": mmseqs_revision},
                }
            }
        )
    )
    monkeypatch.setattr(
        validation, "_target_checks", lambda *args, **kwargs: ([], None, None, None)
    )
    monkeypatch.setattr(validation, "PRODUCTION_STACK", stack)
    monkeypatch.setenv("TREX_FOLDSEEK_BIN", str(executables["foldseek"]))
    monkeypatch.setenv("TREX_MMSEQS_BIN", str(executables["mmseqs"]))

    checks = validation.validate_install(
        target="test",
        enabled_families=(),
        require_backends=True,
        verify_backend_revisions=True,
    )

    assert (
        next(check for check in checks if check.name == "Foldseek version").status
        == "ok"
    )
    assert (
        next(check for check in checks if check.name == "MMseqs2 version").status
        == "ok"
    )


def test_model_preflight_requires_serving_controller_python(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        validation, "_target_checks", lambda *args, **kwargs: ([], None, None, None)
    )
    monkeypatch.delenv("TREX_CONTROLLER_PYTHON", raising=False)

    checks = validation.validate_install(
        target="test", enabled_families=(), require_model=True
    )

    serving = next(
        check for check in checks if check.name == "campaign serving/controller Python"
    )
    assert serving.status == "fail"
