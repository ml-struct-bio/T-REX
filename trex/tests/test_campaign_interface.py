"""Contract tests for the user-facing campaign input/output boundary."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from trex.archive import Archive
from trex.campaign.artifacts import write_campaign_artifacts
from trex.campaign.environment import applied_environment
from trex.campaign.preflight import PreflightReport, preflight_campaign
from trex.campaign.run_lock import campaign_run_lock
from trex.campaign.provenance import (
    CampaignProvenanceArtifact,
    write_campaign_provenance,
)
from trex.campaign.config import ConfigError, load_campaign, parse_campaign
from trex.campaign.status import collect_campaign_status
from trex.cli import main
from trex.provenance import build_model_manifest
from trex.schemas import (
    DispatchRecord,
    EvidenceSummary,
    LaunchDecision,
    LLMHealthSummary,
    ResultRecord,
    RouteHealthSummary,
)


_ONE_RESIDUE_PDB = """\
ATOM      1  N   ALA A   1      11.104  13.207   9.121  1.00 20.00           N
ATOM      2  CA  ALA A   1      12.560  13.207   9.121  1.00 20.00           C
ATOM      3  C   ALA A   1      13.000  14.620   9.500  1.00 20.00           C
TER
END
"""


def _campaign_file(
    root: Path,
    *,
    families: list[str] | None = None,
    extra: dict | None = None,
) -> Path:
    (root / "inputs").mkdir(parents=True, exist_ok=True)
    constraint = root / "inputs" / "target.json"
    pdb = root / "inputs" / "target.pdb"
    constraint.write_text(
        json.dumps(
            {
                "target_id": "custom_001",
                "target_class": "protein",
                "chain_ids": ["A"],
                "hotspots": ["A1"],
                "panel_size_K": 8,
            }
        )
    )
    pdb.write_text(_ONE_RESIDUE_PDB)
    payload = {
        "schema_version": "trex.campaign.v1",
        "name": "readable-test",
        "target": {
            "name": "custom",
            "constraint": "inputs/target.json",
            "pdb": "inputs/target.pdb",
        },
        "run": {
            "archive_root": "runs/campaign-a",
            "max_wall_hours": 2.5,
            "seed": 7,
            "worker_gpus": [1, 2],
            "enabled_families": families or ["complexa_beam"],
        },
        "llm": {"base_url": "http://127.0.0.1:12000/v1"},
    }
    if extra:
        payload.update(extra)
    path = root / "campaign.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


def test_load_campaign_resolves_paths_and_controller_input(tmp_path: Path) -> None:
    source = _campaign_file(tmp_path)

    campaign = load_campaign(source)

    assert campaign.target.target_id == "custom_001"
    assert campaign.config.run.archive_root == (tmp_path / "runs/campaign-a").resolve()
    assert campaign.config.run.worker_gpus == ("1", "2")
    assert campaign.effective_families == ("complexa_beam",)
    argv = campaign.controller_argv()
    assert argv[argv.index("--worker-gpus") + 1] == "1,2"
    assert argv[argv.index("--target-pdb") + 1] == str(
        (tmp_path / "inputs/target.pdb").resolve()
    )
    resolved = campaign.as_dict()
    assert resolved["controller_interface"]["argv"] == argv
    assert resolved["controller_interface"]["runtime_paths"]["af2_data_dir"] == str(
        campaign.runtime_paths().af2_data_dir
    )
    assert resolved["campaign"]["source_sha256"] == campaign.source_sha256


def test_diagnostic_family_makes_canonical_scoring_explicit(tmp_path: Path) -> None:
    campaign = load_campaign(_campaign_file(tmp_path, families=["bindcraft"]))

    assert campaign.effective_families == ("bindcraft", "structure_refilter")
    assert any("canonical AF2 scoring" in note for note in campaign.resolution_notes)


def test_unknown_config_key_fails_with_location(tmp_path: Path) -> None:
    source = _campaign_file(tmp_path, extra={"mystery": True})

    with pytest.raises(ConfigError, match=r"campaign config.*mystery"):
        parse_campaign(source)


def test_duplicate_worker_gpu_fails_before_execution(tmp_path: Path) -> None:
    source = _campaign_file(tmp_path)
    payload = yaml.safe_load(source.read_text())
    payload["run"]["worker_gpus"] = [1, 1]
    source.write_text(yaml.safe_dump(payload))

    with pytest.raises(ConfigError, match="must not contain duplicates"):
        parse_campaign(source)


def test_comma_delimited_list_item_is_rejected_as_ambiguous(tmp_path: Path) -> None:
    source = _campaign_file(tmp_path)
    payload = yaml.safe_load(source.read_text())
    payload["run"]["worker_gpus"] = ["1,2"]
    source.write_text(yaml.safe_dump(payload))

    with pytest.raises(ConfigError, match="one unambiguous item"):
        parse_campaign(source)


def test_duplicate_yaml_key_fails_instead_of_silently_overriding(
    tmp_path: Path,
) -> None:
    source = tmp_path / "campaign.yaml"
    source.write_text(
        """\
schema_version: trex.campaign.v1
name: first
name: second
target: {name: custom, constraint: target.json, pdb: target.pdb}
run: {archive_root: run, enabled_families: [complexa_beam]}
"""
    )

    with pytest.raises(ConfigError, match="duplicate key 'name'"):
        parse_campaign(source)


def test_campaign_artifacts_preserve_exact_and_resolved_input(tmp_path: Path) -> None:
    campaign = load_campaign(_campaign_file(tmp_path))

    artifacts = write_campaign_artifacts(campaign)

    assert artifacts.input_path.read_bytes() == campaign.source_path.read_bytes()
    resolved = json.loads(artifacts.resolved_path.read_text())
    assert resolved["target"]["target_id"] == "custom_001"
    assert resolved["run"]["worker_gpus"] == ["1", "2"]

    first_resolved = artifacts.resolved_path.read_bytes()
    payload = yaml.safe_load(campaign.source_path.read_text())
    payload["name"] = "changed-on-resume"
    campaign.source_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    changed = load_campaign(campaign.source_path)
    changed_artifacts = write_campaign_artifacts(changed)

    assert changed_artifacts.input_path != artifacts.input_path
    assert changed_artifacts.resolved_path != artifacts.resolved_path
    assert changed_artifacts.input_path.stem.removeprefix(
        "campaign_input_"
    ) == changed_artifacts.resolved_path.stem.removeprefix("campaign_resolved_")
    assert artifacts.resolved_path.read_bytes() == first_resolved


def test_campaign_artifact_uses_the_exact_bytes_that_were_parsed(
    tmp_path: Path,
) -> None:
    source = _campaign_file(tmp_path)
    parsed_bytes = source.read_bytes()
    campaign = load_campaign(source)
    source.write_text(source.read_text().replace("readable-test", "edited-later"))

    artifacts = write_campaign_artifacts(campaign)

    assert artifacts.input_path.read_bytes() == parsed_bytes
    resolved = json.loads(artifacts.resolved_path.read_text())
    assert resolved["campaign"]["source_sha256"] == campaign.source_sha256


def test_resolved_backend_defaults_are_absolute_and_executable_symlinks_survive(
    tmp_path: Path,
) -> None:
    source = _campaign_file(tmp_path)
    env_root = tmp_path / "fake-env"
    python_target = env_root / "python-base"
    python_link = env_root / "bin" / "python"
    python_link.parent.mkdir(parents=True)
    python_target.write_text("python")
    python_link.symlink_to(python_target)
    payload = yaml.safe_load(source.read_text())
    payload["backends"] = {"complexa_python": str(python_link)}
    source.write_text(yaml.safe_dump(payload, sort_keys=False))

    campaign = load_campaign(source)

    assert campaign.config.backends.repo_root is not None
    assert campaign.config.backends.repo_root.is_absolute()
    assert campaign.config.backends.complexa_python == python_link
    assert campaign.config.backends.complexa_python.is_symlink()


def test_resolved_environment_records_tuning_but_redacts_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TREX_EXPERIMENT_LABEL", "replicate-a")
    monkeypatch.setenv("TREX_API_KEY", "must-not-be-persisted")

    resolved = load_campaign(_campaign_file(tmp_path)).as_dict()
    inherited = resolved["controller_interface"]["inherited_trex_environment"]

    assert inherited["TREX_EXPERIMENT_LABEL"] == "replicate-a"
    assert inherited["TREX_API_KEY"] == "<redacted>"


def test_dry_run_is_read_only_and_prints_resolved_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _campaign_file(tmp_path)
    archive = tmp_path / "runs/campaign-a"

    rc = main(["campaign", "run", str(source), "--dry-run"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "Target input" in captured.out
    assert "Backend runtime" in captured.out
    assert "AF2 parameters" in captured.out
    assert "Dry run complete" in captured.out
    assert not archive.exists()


def test_status_summarizes_execution_without_raw_jsonl_inspection(
    tmp_path: Path,
) -> None:
    root = tmp_path / "archive"
    archive = Archive(root)
    archive.append(
        ResultRecord(
            result_id="result-1",
            parent_ids=[],
            target_id="custom_001",
            backend_family="complexa_beam",
            runtime_bucket_id="rb1",
            metrics={"pLDDT": 91.0, "iPAE": 0.2, "binder_scRMSD": 1.0},
            metrics_calibrated={},
            route_lineage=["complexa_beam"],
            gpu_h=1.0,
            exit_status="ok",
        )
    )
    archive.append(
        LaunchDecision(
            launch_id="launch-1",
            tick_id="tick-1",
            candidate_id="candidate-1",
            status="launched",
            resource_class_concrete={"class": "low"},
            why="test",
        )
    )
    archive.append(
        DispatchRecord(
            dispatch_id="dispatch-1",
            tick_id="tick-1",
            candidate_id="candidate-1",
            status="started",
            launch_id="launch-1",
            worker_slot="0",
            gpu_id="1",
        )
    )

    status = collect_campaign_status(root)

    assert status.record_counts["result_records.jsonl"] == 1
    assert status.launch_counts == {"launched": 1}
    assert status.dispatch_counts == {"started": 1}
    assert status.dispatch_outcome_counts == {"started": 1}
    assert status.result_counts == {"ok": 1}
    assert status.skipped_records == {}
    assert status.provenance_files == ()
    assert "no run_provenance*.json reproducibility artifact" in status.warnings
    assert status.as_dict()["integrity"]["scope"].startswith("typed read health")


def test_status_uses_provenance_as_a_valid_slurm_input_summary(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    (archive_root / "run_provenance.json").write_text(
        json.dumps(
            {
                "target": {"name": "cd45", "target_id": "05_CD45"},
                "controller": {
                    "max_wall_h": 48.0,
                    "worker_gpus": ["1", "2", "3"],
                    "enabled_families": ["complexa_beam"],
                },
                "runtime": {"slurm_job_name": "trex_cd45"},
            }
        )
    )

    status = collect_campaign_status(archive_root)

    assert status.input_artifact_status == "provenance_only"
    assert status.campaign_input == {
        "name": "trex_cd45",
        "target": "cd45",
        "target_id": "05_CD45",
        "source_sha256": None,
        "max_wall_hours": 48.0,
        "worker_gpus": ["1", "2", "3"],
        "enabled_families": ["complexa_beam"],
    }
    assert "no campaign_resolved*.json input artifact" not in status.warnings
    assert status.as_dict()["input_artifact_status"] == "provenance_only"


def _evidence_summary_for_status() -> EvidenceSummary:
    return EvidenceSummary(
        tick_id="tick-1",
        target_id="custom_001",
        target_class="protein",
        schema_version="test",
        elapsed_wall_h=2.0,
        remaining_wall_h=6.0,
        completed_children=11,
        pending_children=2,
        worker_gpu_h_total=4.5,
        worker_gpu_h_last_3_ticks=1.5,
        strict_count=3,
        global_new_strict=1,
        run_su_count=2,
        run_su_count_delta=1,
        su_per_gpu_h_recent=0.5,
        duplicate_fraction=0.25,
        top_bin_share=0.5,
        axis_stats={},
        joint_patterns=[],
        near_miss_count=1,
        panel_ready_count=2,
        panel_ready_bins_covered=2,
        method_health={},
        route_health=RouteHealthSummary(0, 0, 0, 96, None, None, None),
        llm_health=LLMHealthSummary("model", [], 0.0, 0.0, 0, 0.0),
        state_label="productive",
        examples=[],
        metric_availability={},
        worker_wall_gpu_count=3.0,
        worker_wall_gpu_h_total=6.0,
        run_su_per_worker_wall_gpu_h_total=1.0 / 3.0,
        gpu_h_since_last_su=0.75,
        production_panel_status="ok",
        production_panel_value=0.8,
    )


def test_status_uses_current_evidence_schema_names(tmp_path: Path) -> None:
    archive = Archive(tmp_path / "archive")
    archive.append(_evidence_summary_for_status())

    evidence = collect_campaign_status(archive.root).latest_evidence

    assert evidence is not None
    assert evidence["completed_children"] == 11
    assert evidence["strict_count"] == 3
    assert evidence["run_su_count"] == 2
    assert evidence["worker_wall_gpu_h_total"] == 6.0
    assert evidence["gpu_h_since_last_su"] == 0.75
    assert "n_strict_success" not in evidence
    assert "worker_wall_gpu_h" not in evidence


def test_campaign_provenance_links_input_model_target_and_prompts(
    tmp_path: Path,
) -> None:
    source = _campaign_file(tmp_path)
    model_root = tmp_path / "model"
    model_root.mkdir()
    (model_root / "config.json").write_text("{}\n")
    manifest = build_model_manifest(model_root)
    manifest_path = tmp_path / "model-manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    payload = yaml.safe_load(source.read_text())
    payload["backends"] = {
        "repo_root": str(Path(__file__).resolve().parents[2]),
        "qwen_model_path": str(model_root),
        "qwen_model_manifest": str(manifest_path),
    }
    source.write_text(yaml.safe_dump(payload, sort_keys=False))
    campaign = load_campaign(source)
    artifacts = write_campaign_artifacts(campaign)

    clean_state = {
        "path": str(Path(__file__).resolve().parents[2]),
        "git_head": "test-head",
        "dirty": False,
        "status_sha256": None,
        "tracked_diff_sha256": None,
        "tracked_diff_artifact": None,
        "untracked_paths": [],
        "untracked_path_count": 0,
        "capture_error": None,
        "capture_mode": "normal",
        "tracked_diff_lfs_excludes": [],
    }
    with applied_environment(campaign.environment()), patch(
        "trex.provenance.git_state", return_value=clean_state
    ), patch("trex.provenance.source_tree_digest", return_value=("source-tree-sha", 1)):
        first = write_campaign_provenance(campaign, artifacts)
        second = write_campaign_provenance(campaign, artifacts)

    assert first.path.name == "run_provenance.json"
    assert second.path.name == "run_provenance_resume_001.json"
    record = json.loads(first.path.read_text())
    assert record["campaign"]["source_sha256"] == campaign.source_sha256
    assert record["campaign"]["input_artifact_sha256"] == campaign.source_sha256
    assert record["model"]["content_sha256"] == manifest["content_sha256"]
    assert record["model"]["served_name"] == "Qwen/Qwen3.6-27B-FP8"
    assert record["model"]["client_model"] == "vllm/Qwen/Qwen3.6-27B-FP8"
    planner_call = record["prompts"]["roles"]["planner"]["call_configuration"]
    assert planner_call["model"] == "vllm/Qwen/Qwen3.6-27B-FP8"
    assert planner_call["base_url"] == "http://127.0.0.1:12000/v1"
    assert record["target"]["pdb_sha256"]
    assert set(record["prompts"]["roles"]) == {"planner", "supervisor", "critic"}


def test_preflight_json_output_is_versioned(tmp_path: Path, capsys) -> None:
    source = _campaign_file(tmp_path)
    with patch(
        "trex.campaign.cli.preflight_campaign", return_value=PreflightReport(())
    ):
        assert (
            main(
                [
                    "campaign",
                    "preflight",
                    str(source),
                    "--json",
                ]
            )
            == 0
        )

    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "trex.campaign-preflight.v1"
    assert payload["preflight"]["schema_version"] == "trex.preflight-report.v1"
    assert payload["resolved_campaign"]["schema_version"] == "trex.resolved-campaign.v1"


def test_real_campaign_run_captures_provenance_before_controller(
    tmp_path: Path,
) -> None:
    source = _campaign_file(tmp_path)
    provenance = CampaignProvenanceArtifact(
        path=tmp_path / "run_provenance.json",
        source_tree_sha256="source-sha",
        model_content_sha256="model-sha",
    )
    observed: dict[str, str | None] = {}

    def controller_main(*_args, **_kwargs):
        observed["model_digest"] = os.environ.get("TREX_MODEL_DIGEST")
        return 0

    with patch(
        "trex.campaign.cli.preflight_campaign", return_value=PreflightReport(())
    ) as preflight, patch(
        "trex.campaign.cli.write_campaign_provenance", return_value=provenance
    ) as capture, patch(
        "trex.controller.main", side_effect=controller_main
    ) as controller:
        assert main(["campaign", "run", str(source)]) == 0

    assert preflight.call_args.kwargs["require_model"] is True
    assert capture.call_count == 1
    assert controller.call_count == 1
    assert observed["model_digest"] == "model-sha"


def test_campaign_memory_is_explicit_resolved_and_preflighted(
    tmp_path: Path,
) -> None:
    source = _campaign_file(tmp_path)
    memory_path = tmp_path / "inputs" / "cross-campaign-memory.json"
    memory_path.write_text(
        json.dumps(
            {
                "schema_version": "cross-campaign-memory-v3",
                "scope": {"historical_campaign_count": 1},
                "usage_contract": {"authority": "advisory"},
                "stable_core": {"entries": []},
                "contextual_design_memory": {"entries": []},
            }
        )
    )
    payload = yaml.safe_load(source.read_text())
    payload["memory"] = {"cross_campaign_path": "inputs/cross-campaign-memory.json"}
    source.write_text(yaml.safe_dump(payload, sort_keys=False))

    campaign = load_campaign(source)
    report = preflight_campaign(campaign)

    assert campaign.config.memory.cross_campaign_path == memory_path.resolve()
    assert campaign.environment()["TREX_CROSS_CAMPAIGN_MEMORY"] == str(
        memory_path.resolve()
    )
    assert campaign.as_dict()["memory"]["cross_campaign_path"] == str(
        memory_path.resolve()
    )
    memory_check = next(
        check for check in report.checks if check.name == "cross-campaign memory"
    )
    assert memory_check.status == "ok"
    assert "sha256=" in memory_check.detail


def test_campaign_run_lock_rejects_concurrent_launcher(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"

    with campaign_run_lock(archive_root):
        with pytest.raises(OSError, match="active launcher"):
            with campaign_run_lock(archive_root):
                raise AssertionError("second launcher unexpectedly acquired the lock")


def test_campaign_status_reports_invalid_resolved_input(
    tmp_path: Path,
) -> None:
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    (archive_root / "campaign_resolved.json").write_text("not-json\n")

    status = collect_campaign_status(archive_root)

    assert status.campaign_input is None
    assert any(
        "campaign_resolved.json: invalid resolved input" in warning
        for warning in status.warnings
    )


def test_campaign_status_reports_invalid_resolved_input_section(
    tmp_path: Path,
) -> None:
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    (archive_root / "campaign_resolved.json").write_text('{"campaign": []}\n')

    status = collect_campaign_status(archive_root)

    assert status.campaign_input is None
    assert any(
        "section 'campaign' must be a JSON object" in warning
        for warning in status.warnings
    )
