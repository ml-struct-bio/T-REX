from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from trex.provenance import (
    MODEL_MANIFEST_SCHEMA,
    PROVENANCE_CAPTURE_OUTPUT_SCHEMA_VERSION,
    _run_text,
    build_model_manifest,
    capture_run_provenance,
    git_state,
    main as provenance_main,
    model_digest_from_env,
    source_tree_digest,
    verify_model_manifest,
)


def test_model_manifest_hashes_content_and_detects_size_change(tmp_path: Path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n")
    (model / "weights.safetensors").write_bytes(b"weights")
    manifest = build_model_manifest(model)
    assert manifest["schema_version"] == MODEL_MANIFEST_SCHEMA
    assert manifest["n_files"] == 2
    assert verify_model_manifest(model, manifest) == manifest["content_sha256"]
    assert (
        verify_model_manifest(model, manifest, verify_content=True)
        == manifest["content_sha256"]
    )
    (model / "weights.safetensors").write_bytes(b"changed-size")
    try:
        verify_model_manifest(model, manifest)
    except ValueError as exc:
        assert "size changed" in str(exc)
    else:
        raise AssertionError("changed model size was not rejected")


def test_model_manifest_full_verification_detects_same_size_change(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    weight = model / "weights.bin"
    weight.write_bytes(b"abcd")
    manifest = build_model_manifest(model)
    weight.write_bytes(b"wxyz")

    assert verify_model_manifest(model, manifest) == manifest["content_sha256"]
    with pytest.raises(ValueError, match="content changed"):
        verify_model_manifest(model, manifest, verify_content=True)


def test_source_tree_digest_ignores_caches(tmp_path: Path):
    package = tmp_path / "trex"
    package.mkdir()
    (package / "a.py").write_text("x = 1\n")
    cache = package / "__pycache__"
    cache.mkdir()
    (cache / "a.pyc").write_bytes(b"ignored")
    first, n_first = source_tree_digest(tmp_path)
    (cache / "a.pyc").write_bytes(b"still ignored")
    second, n_second = source_tree_digest(tmp_path)
    assert first == second
    assert n_first == n_second == 1


@pytest.mark.parametrize("launch_file", ["slurm/T-REX.slurm", "scripts/submit.sh"])
def test_source_tree_digest_detects_launch_changes(tmp_path: Path, launch_file: str):
    package = tmp_path / "trex"
    package.mkdir()
    (package / "a.py").write_text("x = 1\n")
    launcher = tmp_path / launch_file
    launcher.parent.mkdir(parents=True, exist_ok=True)
    launcher.write_text("#!/bin/bash\nexit 0\n")
    first, first_count = source_tree_digest(tmp_path)
    launcher.write_text("#!/bin/bash\nexit 1\n")
    second, second_count = source_tree_digest(tmp_path)
    assert first != second
    assert first_count == second_count == 2


def test_capture_writes_atomic_complete_record(tmp_path: Path, monkeypatch):
    source = tmp_path / "source"
    package = source / "trex"
    package.mkdir(parents=True)
    (package / "a.py").write_text("x = 1\n")
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n")
    model_manifest = build_model_manifest(model)
    manifest_path = tmp_path / "model_manifest.json"
    manifest_path.write_text(json.dumps(model_manifest))
    target_pdb = tmp_path / "target.pdb"
    target_pdb.write_text("ATOM\n")
    target_config = tmp_path / "target.json"
    target_config.write_text("{}\n")
    monkeypatch.setenv("TREX_MODEL_DIGEST", model_manifest["content_sha256"])
    monkeypatch.setenv("TREX_COMPLEXA_PYTHON", "/envs/complexa/.venv/bin/python")
    args = argparse.Namespace(
        out=str(tmp_path / "archive" / "run_provenance.json"),
        source_root=str(source),
        model_path=str(model),
        model_manifest=str(manifest_path),
        served_model="test/model",
        target="test",
        target_id="target_id",
        target_pdb=str(target_pdb),
        target_config=str(target_config),
        max_wall_h="47",
        foldseek_su_tm_score="0.6",
        foldseek_collapse_tm_score="0.6",
        enabled_families="complexa_beam,bindcraft",
        worker_gpus="1,2,3",
        charged_gpus="4",
        seed="0",
        critic_enabled="1",
        evidence_skip_enabled="0",
    )
    payload = capture_run_provenance(args)
    archived = json.loads(Path(args.out).read_text())
    assert archived == payload
    assert archived["controller"]["critic_enabled"] is True
    assert archived["prompts"]["schema_version"] == "trex.prompt-catalog.v2"
    assert archived["prompts"]["deterministic_guard"]["enabled"] is True
    critic_metadata = archived["prompts"]["roles"]["critic"]
    assert critic_metadata["usage_scope"] == "standalone_optional_llm"
    assert critic_metadata["call_configuration"]["enabled"] is False
    assert archived["model"]["content_sha256"] == model_manifest["content_sha256"]
    assert archived["target"]["pdb_sha256"]
    assert archived["source"]["tree_sha256"]
    assert (
        archived["runtime"]["af2_proteinmpnn_python"]
        == "/envs/complexa/.venv/bin/python"
    )
    assert model_digest_from_env() == model_manifest["content_sha256"]


def test_capture_cli_emits_versioned_machine_json(capsys) -> None:
    payload = {
        "source": {"tree_sha256": "source-digest"},
        "model": {"content_sha256": "model-digest"},
        "target": {"pdb_sha256": "target-digest"},
    }
    with patch("trex.provenance._parser") as parser_factory, patch(
        "trex.provenance.capture_run_provenance", return_value=payload
    ):
        parser_factory.return_value.parse_args.return_value = argparse.Namespace(
            command="capture"
        )
        assert provenance_main([]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output == {
        "model_content_sha256": "model-digest",
        "schema_version": PROVENANCE_CAPTURE_OUTPUT_SCHEMA_VERSION,
        "source_tree_sha256": "source-digest",
        "target_pdb_sha256": "target-digest",
    }


def test_git_state_archives_staged_and_unstaged_tracked_changes(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=repo, check=True
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    tracked = repo / "tracked.txt"
    tracked.write_text("one\n")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=repo, check=True)
    tracked.write_text("two\n")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
    tracked.write_text("three\n")
    patch = tmp_path / "archive" / "repo.patch"
    state = git_state(repo, diff_output=patch)
    assert state["dirty"] is True
    assert state["tracked_diff_sha256"]
    assert state["tracked_diff_artifact"] == str(patch)
    patch_text = patch.read_text()
    assert "-one" in patch_text
    assert "+three" in patch_text


def test_git_state_fails_closed_when_tracked_state_is_unavailable(tmp_path: Path):
    with patch(
        "trex.provenance._run_text",
        side_effect=["deadbeef", None, None],
    ):
        with pytest.raises(RuntimeError, match="could not capture tracked git status"):
            git_state(tmp_path, strict=True)


def test_git_state_records_capture_error_when_non_strict(tmp_path: Path):
    with patch(
        "trex.provenance._run_text",
        side_effect=["deadbeef", None, None],
    ):
        state = git_state(tmp_path)
    assert state["git_head"] == "deadbeef"
    assert state["dirty"] is None
    assert state["capture_error"].startswith("could not capture tracked git status")


def test_git_state_uses_lfs_excluded_fallback(tmp_path: Path):
    (tmp_path / ".gitattributes").write_text(
        "assets/data/*.csv filter=lfs diff=lfs merge=lfs -text\n"
    )
    patch_file = tmp_path / "patches" / "repo.patch"
    with patch(
        "trex.provenance._run_text",
        side_effect=[
            "deadbeef",
            None,
            None,
            " M configs/targets.yaml",
            "diff --git a/configs/targets.yaml b/configs/targets.yaml\n",
        ],
    ):
        state = git_state(tmp_path, diff_output=patch_file)
    assert state["dirty"] is True
    assert state["capture_error"] is None
    assert state["capture_mode"] == "lfs_excluded"
    assert state["tracked_diff_lfs_excludes"] == [":!assets/data/*.csv"]
    assert state["tracked_diff_artifact"] == str(patch_file)
    assert patch_file.read_text().startswith("diff --git")


def test_git_state_continues_when_lock_dir_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("TREX_PROVENANCE_LOCK_DIR", str(tmp_path / "locks"))
    with patch("trex.provenance.Path.mkdir", side_effect=OSError("read-only"),), patch(
        "trex.provenance._run_text",
        side_effect=["deadbeef", "", ""],
    ):
        state = git_state(tmp_path)
    assert state["git_head"] == "deadbeef"
    assert state["dirty"] is False
    assert state["capture_error"] is None


def test_run_text_retries_transient_failure(tmp_path: Path):
    completed = subprocess.CompletedProcess(["git"], 0, stdout="ok\n", stderr="")
    with patch(
        "trex.provenance.subprocess.run",
        side_effect=[OSError("transient"), completed],
    ) as run, patch("trex.provenance.time.sleep"):
        assert _run_text(["git", "status"], tmp_path) == "ok"
    assert run.call_count == 2


def test_git_state_marks_untracked_files_dirty(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=repo, check=True
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    tracked = repo / "tracked.txt"
    tracked.write_text("tracked\n")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=repo, check=True)
    (repo / "new_source.py").write_text("VALUE = 1\n")

    state = git_state(repo, include_diff_hash=False)

    assert state["dirty"] is True
    assert state["untracked_path_count"] == 1
    assert state["untracked_paths"] == ["new_source.py"]
