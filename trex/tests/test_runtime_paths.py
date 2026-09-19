"""Contracts for the immutable backend runtime-path snapshot."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from trex.campaign import BackendPaths, RuntimePaths
from trex.resource_paths import PACKAGED_DATA_ROOT, publication_data_path
from trex.campaign.runtime import (
    BACKEND_PATH_ENVIRONMENT_VARIABLES,
    resolve_environment_executable_path,
)
from trex.targets import resolve_target


def test_backend_input_and_runtime_environment_names_cannot_drift() -> None:
    assert set(BACKEND_PATH_ENVIRONMENT_VARIABLES) == set(
        BackendPaths.__dataclass_fields__
    )


def test_runtime_paths_resolve_overrides_and_preserve_executable_symlinks(
    tmp_path: Path,
) -> None:
    repository_root = tmp_path / "trex-repository"
    legacy_complexa_repository = tmp_path / "legacy-complexa"
    python_target = tmp_path / "python-base"
    python_target.write_text("python")
    complexa_python = tmp_path / "complexa-env" / "bin" / "python"
    complexa_python.parent.mkdir(parents=True)
    complexa_python.symlink_to(python_target)
    foldseek_binary = tmp_path / "tools" / "foldseek"

    runtime_paths = RuntimePaths.from_environment(
        {
            "PATH": "",
            "TREX_REPO_ROOT": str(repository_root),
            "TREX_LEGACY_COMPLEXA_REPO": str(
                legacy_complexa_repository
            ),
            "TREX_COMPLEXA_PYTHON": str(complexa_python),
            "TREX_FOLDSEEK_BIN": str(foldseek_binary),
        }
    )

    assert runtime_paths.repo_root == repository_root.resolve()
    assert runtime_paths.complexa_python == complexa_python
    assert runtime_paths.complexa_python.is_symlink()
    assert runtime_paths.community_models_root == (
        legacy_complexa_repository.resolve() / "community_models"
    )
    assert runtime_paths.af2_data_dir == (
        runtime_paths.community_models_root / "ckpts" / "AF2"
    )
    assert runtime_paths.proteinmpnn_weights == (
        runtime_paths.proteinmpnn_dir
        / "vanilla_model_weights"
        / "v_48_020.pt"
    )
    assert runtime_paths.foldseek_command == str(foldseek_binary)
    assert runtime_paths.mmseqs_command == "mmseqs"


def test_backend_paths_environment_round_trips_into_runtime_paths(
    tmp_path: Path,
) -> None:
    configured_paths = BackendPaths(
        repo_root=tmp_path / "repo",
        external_root=tmp_path / "external",
        complexa_repo=tmp_path / "complexa",
        legacy_complexa_repo=tmp_path / "legacy-complexa",
        complexa_python=tmp_path / "complexa-env/bin/python",
        bindcraft_repo=tmp_path / "bindcraft",
        bindcraft_env=tmp_path / "bindcraft-env",
        boltzgen_repo=tmp_path / "boltzgen",
        boltzgen_binary=tmp_path / "boltzgen-env/bin/boltzgen",
        boltzgen_cache=tmp_path / "boltzgen-cache",
        foldseek_binary=tmp_path / "tools/foldseek",
        mmseqs_binary=tmp_path / "tools/mmseqs",
        qwen_model_path=tmp_path / "models/qwen",
        qwen_model_manifest=tmp_path / "qwen-manifest.json",
    )

    runtime_paths = configured_paths.to_runtime_paths()

    for field_name in BackendPaths.__dataclass_fields__:
        expected_path = getattr(configured_paths, field_name)
        assert expected_path is not None
        actual_path = getattr(runtime_paths, field_name)
        if field_name in {
            "complexa_python",
            "boltzgen_binary",
            "foldseek_binary",
            "mmseqs_binary",
        }:
            assert actual_path == expected_path
        else:
            assert actual_path == expected_path.resolve()


def test_runtime_path_snapshot_is_immutable_and_inspectable(tmp_path: Path) -> None:
    runtime_paths = RuntimePaths.from_environment(
        {"PATH": ""},
        default_repo_root=tmp_path,
    )

    with pytest.raises(FrozenInstanceError):
        runtime_paths.repo_root = tmp_path / "changed"  # type: ignore[misc]

    snapshot = runtime_paths.as_dict()
    assert snapshot["repo_root"] == str(tmp_path.resolve())
    assert snapshot["af2_data_dir"] == str(runtime_paths.af2_data_dir)
    assert snapshot["proteinmpnn_weights"] == str(
        runtime_paths.proteinmpnn_weights
    )


def test_executable_path_helper_does_not_dereference_venv_python(
    tmp_path: Path,
) -> None:
    target = tmp_path / "python-base"
    target.write_text("python")
    symlink = tmp_path / ".venv" / "bin" / "python"
    symlink.parent.mkdir(parents=True)
    symlink.symlink_to(target)

    resolved = resolve_environment_executable_path(
        "TREX_COMPLEXA_PYTHON",
        "/unused",
        environment={"TREX_COMPLEXA_PYTHON": str(symlink)},
    )

    assert resolved == symlink
    assert resolved != resolved.resolve()


def test_packaged_publication_data_falls_back_outside_a_checkout(
    tmp_path: Path,
) -> None:
    path = publication_data_path(
        tmp_path / "not-a-checkout", "config/targets/registry.json"
    )

    assert path == PACKAGED_DATA_ROOT / "config/targets/registry.json"
    assert path.is_file()
    assert path.read_bytes() == (
        Path(__file__).resolve().parents[2] / "config/targets/registry.json"
    ).read_bytes()


def test_registered_target_uses_packaged_constraint_outside_a_checkout(
    tmp_path: Path,
) -> None:
    pdb_path = tmp_path / "assets" / "bindcraft_targets" / "CD45.pdb"
    pdb_path.parent.mkdir(parents=True)
    pdb_path.write_text("HEADER    TEST TARGET\n")

    target = resolve_target(
        "cd45",
        repo_root=tmp_path / "not-a-checkout",
        asset_root=tmp_path / "assets",
    )

    assert target.registered
    assert target.target_id == "05_CD45"
    assert target.config_path == PACKAGED_DATA_ROOT / "config/targets/cd45.json"
    assert target.pdb_path == pdb_path
