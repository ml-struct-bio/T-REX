"""Production-profile and original shutdown-compatibility regressions."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

from trex.campaign.config import parse_campaign
from trex.campaign.controller_args import parse_controller_args
from trex.campaign.models import PolicyConfig, RunConfig
from trex.campaign.template import campaign_template


ROOT = Path(__file__).resolve().parents[2]


def _parse_payload(tmp_path, payload):
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump(payload))
    return parse_campaign(path)


@pytest.fixture
def verifier():
    spec = importlib.util.spec_from_file_location(
        "execution_parity", ROOT / "scripts/verify_execution_parity.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def execution_trees(tmp_path, verifier):
    source = tmp_path / "original"
    release = tmp_path / "release"
    (source / verifier.SOURCE_PACKAGE).mkdir(parents=True)
    (source / "slurm").mkdir()
    (release / "trex/execution").mkdir(parents=True)
    (release / "slurm").mkdir()
    profile = (
        "#SBATCH --time=48:00:00\n"
        "#SBATCH --gres=gpu:h100:4\n"
        "#SBATCH --cpus-per-task=16\n"
        "#SBATCH --mem=192G\n"
        'MAX_WALL_H="${MAX_WALL_H:-47.0}"\n'
    )
    (source / "slurm/v7_3_3_per_target_node.slurm").write_text(profile)
    (release / "slurm/T-REX.slurm").write_text(
        profile.replace("MAX_WALL_H", "TREX_MAX_WALL_H")
        .replace("48:00:00", "49:00:00")
        .replace("47.0", "48.0")
    )
    original = """
def main():
    drain_grace_per_slot_s = 600.0
    for slot in pool:
        if not slot.busy:
            continue
        try:
            rc = slot.proc.wait(timeout=drain_grace_per_slot_s)
        except subprocess.TimeoutExpired:
            elapsed_slot_s = time.time() - slot.launched_at
            try:
                _parse_and_archive_slot(
                    slot, rc=-1, archive=archive, target=target,
                    chain_seq_ref=chain_seq_ref,
                    elapsed_gpu_h=elapsed_slot_s / 3600.0,
                )
            except Exception as exc:
                _append_parse_failed_dispatch_record(
                    archive, slot,
                    f"drain-timeout parse/archive error: {type(exc).__name__}: {exc}",
                    target_id=target.target_id,
                )
            _terminate_process_group(slot.proc)
            continue
        try:
            elapsed_slot_s = time.time() - slot.launched_at
            _parse_and_archive_slot(
                slot, rc=rc, archive=archive, target=target,
                chain_seq_ref=chain_seq_ref,
                elapsed_gpu_h=elapsed_slot_s / 3600.0,
            )
        except Exception as exc:
            _append_parse_failed_dispatch_record(
                archive, slot,
                f"drain parse/archive error: {type(exc).__name__}: {exc}",
                target_id=target.target_id,
            )
        slot.free()
"""
    (source / verifier.SOURCE_PACKAGE / "phase2_v7_controller.py").write_text(original)
    (release / "trex/controller.py").write_text(
        "def main():\n    drain_worker_slots(pool, grace_seconds_per_slot=600.0)\n"
    )
    (release / "trex/execution/worker_supervision.py").write_text(
        (ROOT / "trex/execution/worker_supervision.py").read_text()
    )
    return source, release


def test_shutdown_matches_original_per_slot_trace(verifier, execution_trees):
    source, release = execution_trees
    report = verifier.compare_execution(source, release)
    assert report["passed"], report
    assert report["execution_equal"] is False
    assert report["shutdown_equal"] is True
    assert report["launch_profile"]["equal"] is False
    assert report["launch_profile"]["matches_documented_profiles"] is True
    assert len(report["shutdown_cases"]) == 8
    mixed = report["shutdown_cases"][-1]["public"]
    waits = [event for event in mixed["events"] if event[0] == "wait"]
    assert [event[2] for event in waits] == [600.0] * 4
    assert mixed["busy_after"] == [False, False, True, False, True]


@pytest.mark.parametrize(
    "relative,before,after",
    [
        ("trex/controller.py", "600.0", "300.0"),
        (
            "trex/execution/worker_supervision.py",
            "timeout=grace_seconds_per_slot",
            "timeout=grace_seconds_per_slot / 2",
        ),
        (
            "trex/execution/worker_supervision.py",
            "            dependencies.terminate_worker_process(worker_slot.proc)\n"
            "            # Match the original final drain",
            "            # Match the original final drain",
        ),
        (
            "trex/execution/worker_supervision.py",
            "            # Match the original final drain",
            "            worker_slot.free()\n"
            "            # Match the original final drain",
        ),
        ("slurm/T-REX.slurm", "49:00:00", "48:00:00"),
        ("slurm/T-REX.slurm", "gpu:h100:4", "gpu:4"),
        ("slurm/T-REX.slurm", "48.0", "47.0"),
    ],
)
def test_execution_parity_detects_regressions(
    verifier,
    execution_trees,
    relative,
    before,
    after,
):
    source, release = execution_trees
    path = release / relative
    text = path.read_text()
    assert before in text
    path.write_text(text.replace(before, after))
    assert not verifier.compare_execution(source, release)["passed"]


def test_execution_parity_fails_closed_for_missing_source(tmp_path, verifier):
    report = verifier.compare_execution(tmp_path / "missing", ROOT)
    assert not report["passed"]
    assert "FileNotFoundError" in report["error"]


def test_new_campaign_defaults_match_production_profile(tmp_path):
    config = _parse_payload(
        tmp_path,
        {
            "schema_version": "trex.campaign.v1",
            "target": {"name": "custom"},
            "run": {"archive_root": "archive"},
        },
    )
    template = campaign_template(
        name="test",
        target="custom",
        archive_root="archive",
        asset_root=None,
        target_constraint=None,
        target_pdb=None,
    )
    example = yaml.safe_load((ROOT / "examples/campaign.yaml").read_text())
    assert config.run.max_wall_hours == RunConfig.max_wall_hours == 48.0
    assert (
        config.policy.selector_mode_window_k == PolicyConfig.selector_mode_window_k == 1
    )
    assert config.policy.selector_adaptive_mode_window_k is False
    assert config.policy.evidence_skip is False
    for payload in (template, example):
        assert payload["run"]["max_wall_hours"] == 48.0
        assert payload["policy"]["selector_mode_window_k"] == 1
        assert payload["policy"]["selector_adaptive_mode_window_k"] is False
        assert payload["policy"]["evidence_skip"] is False


@pytest.mark.parametrize("hours", [0.5, 47.0, 48.0, 72.0])
def test_explicit_existing_campaign_values_are_preserved(tmp_path, hours):
    config = _parse_payload(
        tmp_path,
        {
            "schema_version": "trex.campaign.v1",
            "target": {"name": "custom"},
            "run": {"archive_root": "archive", "max_wall_hours": hours},
            "policy": {
                "selector_mode_window_k": 10,
                "selector_adaptive_mode_window_k": True,
            },
        },
    )
    assert config.run.max_wall_hours == hours
    assert config.policy.selector_mode_window_k == 10
    assert config.policy.selector_adaptive_mode_window_k is True


def test_low_level_controller_keeps_original_cli_defaults(tmp_path):
    args = parse_controller_args(
        [
            "--archive-root",
            str(tmp_path / "archive"),
            "--target-constraint",
            str(tmp_path / "target.json"),
            "--target-pdb",
            str(tmp_path / "target.pdb"),
        ]
    )
    assert args.max_wall_h == 48.0
    assert args.enable_evidence_skip == 0
    assert args.selector_mode_window_k == 10
    assert args.selector_adaptive_mode_window_k == 1


def test_shell_and_environment_template_use_production_window():
    shell = (ROOT / "scripts/run_controller.sh").read_text()
    slurm = (ROOT / "slurm/T-REX.slurm").read_text()
    environment = (ROOT / ".env.example").read_text()
    assert shell.count("${TREX_MAX_WALL_H:-48.0}") == 2
    assert "${TREX_MAX_WALL_H:-48.0}" in slurm
    assert "TREX_MAX_WALL_H=48.0" in environment


@pytest.mark.parametrize("hours", [float("nan"), float("inf"), -float("inf"), 10**400])
def test_campaign_rejects_nonfinite_time_limits(tmp_path, hours):
    with pytest.raises(ValueError, match="run.max_wall_hours must be finite"):
        _parse_payload(
            tmp_path,
            {
                "schema_version": "trex.campaign.v1",
                "target": {"name": "custom"},
                "run": {"archive_root": "archive", "max_wall_hours": hours},
            },
        )


def test_execution_comparison_rejects_changed_source_profile(verifier, execution_trees):
    source, release = execution_trees
    path = source / "slurm/v7_3_3_per_target_node.slurm"
    path.write_text(path.read_text().replace("47.0", "48.0"))
    report = verifier.compare_execution(source, release)
    assert report["passed"] is False
    assert report["launch_profile"]["matches_documented_profiles"] is False
