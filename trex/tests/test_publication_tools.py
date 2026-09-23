from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

from trex.analysis import (
    campaign_summary,
    decision_trace,
    main as analysis_main,
    validate_archive,
)
from trex.archive import Archive
from trex.schemas import ActionCandidate, FeasibilityCheck, PanelSelection, ResultRecord
from trex.archive_schema import archive_layout


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _archive_result(result_id: str, parent_ids: list[str]) -> ResultRecord:
    return ResultRecord(
        result_id=result_id,
        parent_ids=list(parent_ids),
        target_id="target",
        backend_family="test_backend",
        runtime_bucket_id="runtime",
        metrics={},
        metrics_calibrated={},
        route_lineage=[],
        gpu_h=0.0,
        exit_status="ok",
    )


def test_campaign_summary_uses_worker_wall_denominator(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "result_records.jsonl",
        [
            {
                "result_id": "r1",
                "target_id": "T",
                "backend_family": "complexa_beam",
                "exit_status": "ok",
            }
        ],
    )
    _write_jsonl(
        tmp_path / "evidence_summaries.jsonl",
        [
            {
                "tick_id": "tick_001",
                "target_id": "T",
                "state_label": "productive",
                "strict_count": 8,
                "run_su_count": 6,
                "worker_wall_gpu_count": 3,
                "worker_wall_gpu_h_total": 12.0,
                "worker_gpu_h_total": 7.0,
                "charged_gpu_h_total": 16.0,
                "foldseek_su_status": "ok",
                "foldseek_su_coverage": 1.0,
                "structure_dedup_scope": "binder_chain",
            }
        ],
    )
    _write_jsonl(
        tmp_path / "llm_call_records.jsonl",
        [
            {
                "role": "planner",
                "tokens_in": 100,
                "tokens_out": 20,
                "latency_s": 1.5,
                "parse_status": "ok",
                "fallback_triggered": False,
            }
        ],
    )
    _write_jsonl(
        tmp_path / "dispatch_records.jsonl",
        [
            {
                "candidate_id": "c1",
                "status": "started",
                "method_family": "complexa_beam",
                "supervisor_mode": "exploit",
            }
        ],
    )
    summary = campaign_summary(tmp_path)
    assert summary["schema_version"] == "trex.analysis-summary.v1"
    assert summary["endpoint"]["structure_unique_successes_tm_live"] == 6
    assert summary["budget"]["worker_wall_gpu_h"] == 12.0
    assert summary["endpoint"]["su_per_worker_wall_gpu_h"] == 0.5
    assert "charged_gpu_h_informational" not in summary["budget"]
    assert summary["llm_usage"]["tokens_in"] == 100


def test_campaign_summary_distinguishes_raw_and_operator_dispatch_outcomes(
    tmp_path: Path,
) -> None:
    _write_jsonl(
        tmp_path / "dispatch_records.jsonl",
        [
            {"dispatch_id": "started", "status": "started"},
            {"dispatch_id": "parse", "status": "parse_failed"},
            {
                "dispatch_id": "stale_prefetch_tick_candidate",
                "status": "dispatch_failed",
                "why": "stale scientific prefetch: evidence changed",
            },
            {
                "dispatch_id": "dispatch_deferred_tick_candidate",
                "status": "dispatch_failed",
                "why": "high_cost_inflight_cap: one high-cost action is active",
            },
            {
                "dispatch_id": "backend_launch_failure",
                "status": "dispatch_failed",
                "why": "OSError: executable unavailable",
            },
        ],
    )

    summary = campaign_summary(tmp_path)

    assert summary["dispatch_status"] == {
        "dispatch_failed": 3,
        "parse_failed": 1,
        "started": 1,
    }
    assert summary["dispatch_outcome"] == {
        "cancelled_before_start": 1,
        "capacity_deferred": 1,
        "dispatch_failed": 1,
        "parse_failed": 1,
        "started": 1,
    }


def test_archive_validation_accepts_system_origins_but_warns_on_unknown_hypothesis(
    tmp_path: Path,
) -> None:
    feasibility = FeasibilityCheck(True, "runtime", True, True, True, True)
    archive = Archive(tmp_path)
    archive.append(
        ActionCandidate(
            candidate_id="system-candidate",
            hypothesis_ids=[
                "warmstart",
                "evidence_fallback",
                "evidence_fallback_dead_probe",
                "diagnostic_i4_mcts",
                "route_value_replay",
                "cross_family_escape",
            ],
            parent_result_id=None,
            method_family="test_backend",
            operator_id="test_operator",
            lane_id="test_lane",
            config_delta={},
            downstream_route_plan=[],
            estimated_cost_class="low",
            expected_signal="system origin",
            evidence_refs=[],
            feasibility=feasibility,
        )
    )
    archive.append(
        ActionCandidate(
            candidate_id="broken-candidate",
            hypothesis_ids=["missing-hypothesis"],
            parent_result_id=None,
            method_family="test_backend",
            operator_id="test_operator",
            lane_id="test_lane",
            config_delta={},
            downstream_route_plan=[],
            estimated_cost_class="low",
            expected_signal="broken reference",
            evidence_refs=[],
            feasibility=feasibility,
        )
    )

    report = validate_archive(tmp_path)
    hypothesis_warnings = [
        warning for warning in report["warnings"] if "unknown hypothesis_id" in warning
    ]

    assert hypothesis_warnings == [
        "action_candidates.jsonl references unknown hypothesis_id values: "
        "['missing-hypothesis']"
    ]


def test_archive_validation_detects_duplicate_result_ids(tmp_path: Path) -> None:
    row = {"result_id": "same", "target_id": "T"}
    _write_jsonl(tmp_path / "result_records.jsonl", [row, row])
    report = validate_archive(tmp_path)
    assert not report["ok"]
    assert any("duplicate result_id" in item for item in report["errors"])


def test_archive_validation_accepts_minimal_typed_archive(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "result_records.jsonl",
        [
            {
                "result_id": "r1",
                "parent_ids": [],
                "target_id": "T",
                "backend_family": "complexa_beam",
                "runtime_bucket_id": "rb1",
                "metrics": {},
                "metrics_calibrated": {},
                "route_lineage": [],
                "gpu_h": 0.0,
                "exit_status": "ok",
            }
        ],
    )
    (tmp_path / "run_provenance.json").write_text(
        json.dumps(
            {
                "source": {"tree_sha256": "a"},
                "target": {"pdb_sha256": "b"},
                "model": {"content_sha256": "c"},
            }
        )
    )
    report = validate_archive(tmp_path)
    assert report["ok"], report
    assert report["typed_record_counts"]["result_records.jsonl"] == 1


def test_decision_trace_joins_auditable_fields(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "evidence_summaries.jsonl",
        [
            {
                "tick_id": "tick_001",
                "target_id": "T",
                "state_label": "near_miss",
                "diagnostic_driver_tldr": "iPAE limiting",
                "strict_count": 0,
                "run_su_count": 0,
            }
        ],
    )
    _write_jsonl(
        tmp_path / "hypothesis_cards.jsonl",
        [
            {
                "tick_created": 1,
                "hypothesis_id": "h1",
                "claim": "Improve the interface",
                "mode_affinity": {"rescue": 1.0},
                "evidence_refs": ["e1"],
                "recommended_action_families": ["proteinmpnn_redesign"],
                "reasoning_trace": {
                    "observed_signal": "iPAE limiting",
                    "inference": "interface packing is weak",
                    "action_implication": "redesign interface sequence",
                },
            }
        ],
    )
    _write_jsonl(
        tmp_path / "action_candidates.jsonl",
        [
            {
                "candidate_id": "c1",
                "method_family": "proteinmpnn_redesign",
                "operator_id": "redesign",
                "supervisor_mode": "rescue",
                "config_delta": {"sampling_temp": 0.1},
                "expected_signal": "lower iPAE",
            }
        ],
    )
    _write_jsonl(
        tmp_path / "supervisor_decisions.jsonl",
        [
            {
                "tick_id": "tick_001",
                "mode_mixture": {"rescue": 1.0},
                "fallback_used": False,
            }
        ],
    )
    _write_jsonl(
        tmp_path / "launch_decisions.jsonl",
        [
            {
                "tick_id": "tick_001",
                "candidate_id": "c1",
                "status": "launched",
                "why": "criterion-specific rescue",
            }
        ],
    )
    _write_jsonl(
        tmp_path / "dispatch_records.jsonl",
        [
            {
                "tick_id": "tick_001",
                "candidate_id": "c1",
                "status": "started",
                "method_family": "proteinmpnn_redesign",
                "supervisor_mode": "rescue",
            }
        ],
    )
    trace = decision_trace(tmp_path, limit=1)
    assert trace[0]["schema_version"] == "trex.decision-trace-entry.v1"
    assert trace[0]["diagnostic_driver"] == "iPAE limiting"
    assert trace[0]["hypotheses"][0]["hypothesis_id"] == "h1"
    assert trace[0]["launches"][0]["family"] == "proteinmpnn_redesign"
    assert trace[0]["dispatches"][0]["outcome"] == "started"


def test_local_markdown_links_resolve() -> None:
    root = Path(__file__).resolve().parents[2]
    missing: list[str] = []
    documents = list(root.glob("*.md")) + [root / "external/README.md"]
    for directory in ("docs", "examples", "benchmarks", ".github"):
        documents.extend((root / directory).rglob("*.md"))
    for document in documents:
        for destination in re.findall(r"\]\(([^)]+)\)", document.read_text()):
            target, _, anchor = destination.strip().partition("#")
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            resolved = (document.parent / target).resolve() if target else document
            if not resolved.exists():
                missing.append(f"{document.relative_to(root)} -> {destination}")
                continue
            if anchor and resolved.suffix == ".md":
                anchors: set[str] = set()
                occurrences: dict[str, int] = {}
                in_fence = False
                for line in resolved.read_text().splitlines():
                    if line.lstrip().startswith(("```", "~~~")):
                        in_fence = not in_fence
                    if in_fence:
                        continue
                    heading = re.match(r"^#{1,6}\s+(.+?)\s*#*\s*$", line)
                    if heading:
                        slug = re.sub(r"[^\w\- ]", "", heading[1].lower()).replace(" ", "-")
                        count = occurrences.get(slug, 0)
                        occurrences[slug] = count + 1
                        anchors.add(f"{slug}-{count}" if count else slug)
                if anchor not in anchors:
                    missing.append(f"{document.relative_to(root)} -> missing anchor {destination}")
    assert missing == []


def test_distribution_hygiene_rejects_cached_external_and_private_members() -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts/check_distribution.py"
    spec = importlib.util.spec_from_file_location("trex_distribution_check", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    allowed = [".env.example", "external/README.md", "external/patches/dependencies.patch",
               "trex/data/external/patches/dependencies.patch", "trex/tests/test_archive.py"]
    forbidden = ["internal/review.md", "audit/run.log", "reviews/notes.md",
                 "docs/internal/checks.md", "external/BindCraft/README.md", "external/OtherBackend/docs/setup.md",
                 ".env", ".env.assets", ".venv-serving/bin/python", "slurm_logs/job.out",
                 "trex/__pycache__/archive.pyc", "../private.key"]
    assert module.unexpected_members(allowed, wheel=False) == []
    assert module.unexpected_members(allowed + forbidden, wheel=False) == forbidden
    assert module.unexpected_members(allowed, wheel=True) == ["trex/tests/test_archive.py"]


def test_behavioral_parity_approved_divergences_are_fail_closed() -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "verify_behavioral_parity.py"
    spec = importlib.util.spec_from_file_location("trex_parity_verifier", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    approved = module.APPROVED_STANDALONE_DIVERGENCES
    assert set(approved) == {
        "diagnosis_outcome.py",
        "candidate_builder.py",
        "critic_guard.py",
        "lifecycle.py",
        "prompts.py",
        "unified_reasoner.py",
        "foldseek_clusterer.py",
        "output_parsers/af2_refilter.py",
        "sequence_clusterer.py",
        "capability_registry.py",
        "critic.py",
        "evidence_reducer.py",
        "fallback.py",
        "live_tick.py",
        "panel.py",
        "planner.py",
        "refilter_roles.py",
        "schemas.py",
        "selector.py",
        "success_criteria.py",
        "supervisor.py",
    }
    for name, expected in approved.items():
        status, reason = module._ast_comparison_status(
            name,
            expected["source_ast_sha256"],
            expected["trex_ast_sha256"],
        )
        assert status == "approved_standalone_divergence"
        assert reason == expected["reason"]
        changed_status, _ = module._ast_comparison_status(
            name,
            expected["source_ast_sha256"],
            "0" * 64,
        )
        assert changed_status == "unexpected_difference"

    exact_status, exact_reason = module._ast_comparison_status(
        "success_criteria.py", "a" * 64, "a" * 64
    )
    assert exact_status == "exact"
    assert exact_reason is None


def test_parity_snapshot_normalization_drops_only_declared_legacy_families() -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "verify_behavioral_parity.py"
    spec = importlib.util.spec_from_file_location(
        "trex_parity_snapshot_normalizer", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    active = {"family": "active", "availability": "available"}
    legacy = {"family": "old", "availability": "legacy_archive_only"}
    source = {
        "registry": {"active": active, "old": legacy},
        "rows": [active, legacy],
        "policy": {"value": 1},
    }
    release = {
        "registry": {"active": active},
        "rows": [active],
        "policy": {"value": 1},
    }
    (
        normalized_source,
        normalized_release,
        ignored,
    ) = module._normalize_legacy_archive_only_families(source, release)
    assert ignored == ["old"]
    assert normalized_source == normalized_release

    reintroduced = {
        **release,
        "registry": {
            **release["registry"],
            "old": {**legacy, "availability": "available"},
        },
        "rows": [active, {**legacy, "availability": "available"}],
    }
    (
        normalized_source,
        normalized_release,
        ignored,
    ) = module._normalize_legacy_archive_only_families(source, reintroduced)
    assert ignored == []
    assert normalized_source != normalized_release

    missing_active = {"registry": {}, "rows": [], "policy": {"value": 1}}
    (
        normalized_source,
        normalized_release,
        _,
    ) = module._normalize_legacy_archive_only_families(source, missing_active)
    assert normalized_source != normalized_release

    changed_release = {**release, "policy": {"value": 2}}
    (
        normalized_source,
        normalized_release,
        _,
    ) = module._normalize_legacy_archive_only_families(source, changed_release)
    assert normalized_source != normalized_release


@pytest.mark.parametrize(
    "before,after",
    [
        ("if queue_room > 0:", "if new_result and queue_room > 0:"),
        (
            "        sleep_s =",
            "        if not pending:\n            break\n        sleep_s =",
        ),
        ("v7_r{round_id:03d}_", "archive_prefix_v7_r{round_id:03d}_"),
        ("elapsed_h >= args.max_wall_h", "elapsed_h >= args.max_wall_h / 2"),
    ],
)
def test_controller_contract_detects_behavior_changes(tmp_path, before, after):
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "trex_parity_verifier", root / "scripts/verify_behavioral_parity.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "source"
    release = tmp_path / "release"
    source.mkdir()
    release.mkdir()
    baseline = """
def main():
    while True:
        if elapsed_h >= args.max_wall_h:
            break
        n_busy = sum(1 for slot in pool if slot.busy)
        queue_room = max(0, len(pool) - len(pending))
        planned = 0
        if queue_room > 0:
            round_id += 1
        if pending:
            dispatched_post = dispatch()
        else:
            dispatched_post = 0
        sleep_s = _controller_sleep_seconds()
        time.sleep(sleep_s)

def _dispatch_candidate_to_gpu():
    cid_safe = cand.candidate_id.replace("/", "_")
    run_name = f"v7_r{round_id:03d}_{cid_safe}"
"""
    (source / "phase2_v7_controller.py").write_text(baseline)
    (release / "controller.py").write_text(baseline)
    assert module._controller_contract_parity(source, release)["passed"]
    assert before in baseline
    (release / "controller.py").write_text(baseline.replace(before, after))
    report = module._controller_contract_parity(source, release)
    assert report["passed"] is False
    assert any(not section["equal"] for section in report["sections"])


def test_controller_contract_fails_closed_when_source_is_missing(tmp_path):
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "trex_parity_verifier", root / "scripts/verify_behavioral_parity.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module._controller_contract_parity(tmp_path, tmp_path)
    assert report["passed"] is False
    assert "FileNotFoundError" in report["error"]


def test_parity_snapshot_exercises_accounting_and_posthoc_defaults(tmp_path):
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "trex_parity_verifier", root / "scripts/verify_behavioral_parity.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    snapshot, run = module._snapshot(
        python=sys.executable,
        root=root,
        package="trex",
        output_dir=tmp_path,
        name="test_snapshot",
    )
    assert run["returncode"] == 0, run["stderr_tail"]
    assert snapshot["evidence_skip_enabled_by_default"] is False
    assert snapshot["posthoc_collapse_default"] == 0.80
    assert len(snapshot["accounting_evidence_and_prompt"]) == 3
    audit = snapshot["accounting_evidence_and_prompt"][0]
    assert audit["evidence"]["run_su_per_charged_gpu_h_total"] == 0.5
    assert (
        audit["prompt"]["objective_summary"]["charged_gpu_audit"]["role"]
        == "audit_only_not_decision_objective"
    )


def test_archive_validation_rejects_incomplete_current_prompt_provenance(
    tmp_path: Path,
) -> None:
    (tmp_path / "run_provenance.json").write_text(
        json.dumps(
            {
                "schema_version": "v7.3.3_run_provenance_v2",
                "source": {"tree_sha256": "source"},
                "target": {"pdb_sha256": "target"},
                "model": {"content_sha256": "model"},
                "prompts": {
                    "roles": {
                        "planner": {"system_prompt_sha256": "a" * 64},
                    }
                },
            }
        )
    )

    report = validate_archive(tmp_path)

    assert not report["ok"]
    assert any(
        "prompts.roles.supervisor.system_prompt_sha256" in error
        for error in report["errors"]
    )
    assert any(
        "prompts.roles.critic.system_prompt_sha256" in error
        for error in report["errors"]
    )


def test_archive_validation_keeps_v1_provenance_backward_compatible(
    tmp_path: Path,
) -> None:
    (tmp_path / "run_provenance.json").write_text(
        json.dumps(
            {
                "schema_version": "v7.3.3_run_provenance_v1",
                "source": {"tree_sha256": "source"},
                "target": {"pdb_sha256": "target"},
                "model": {"content_sha256": "model"},
            }
        )
    )

    report = validate_archive(tmp_path)

    assert not any("prompts.roles" in error for error in report["errors"])


def test_archive_validation_rejects_unknown_provenance_schema(
    tmp_path: Path,
) -> None:
    (tmp_path / "run_provenance.json").write_text(
        json.dumps(
            {
                "schema_version": "v7.3.3_run_provenance_v999",
                "source": {"tree_sha256": "source"},
                "target": {"pdb_sha256": "target"},
                "model": {"content_sha256": "model"},
            }
        )
    )

    report = validate_archive(tmp_path)

    assert not report["ok"]
    assert any("unsupported provenance schema" in error for error in report["errors"])


def test_archive_validation_handles_non_object_provenance(tmp_path: Path) -> None:
    (tmp_path / "run_provenance.json").write_text("[]\n")

    report = validate_archive(tmp_path)

    assert not report["ok"]
    assert any("provenance root" in error for error in report["errors"])


def test_archive_layout_exposes_fields_and_join_keys() -> None:
    payload = archive_layout()
    streams = {row["file_name"]: row for row in payload["streams"]}

    assert payload["schema_version"] == "trex.archive-layout.v1"
    assert "polymorphic" in payload["lineage_semantics"]
    assert "candidate_id" in payload["lineage_semantics"]
    assert streams["runtime_buckets.jsonl"]["primary_key"] == ["bucket_id"]
    assert streams["launch_decisions.jsonl"]["join_keys"] == [
        "tick_id",
        "candidate_id",
        "launch_id",
    ]
    result_fields = {
        row["name"]: row for row in streams["result_records.jsonl"]["fields"]
    }
    assert result_fields["result_id"]["required"] is True
    assert result_fields["bins"]["required"] is False
    assert {
        "source_stream": "dispatch_records.jsonl",
        "source_field": "launch_id",
        "target_stream": "launch_decisions.jsonl",
        "target_field": "launch_id",
    } in payload["relationships"]
    assert {
        "source_stream": "result_records.jsonl",
        "source_field": "parent_ids",
        "target_stream": "action_candidates.jsonl",
        "target_field": "candidate_id",
    } in payload["relationships"]
    fields_by_stream = {
        name: {field["name"] for field in stream["fields"]}
        for name, stream in streams.items()
    }
    for stream in streams.values():
        for field_name in stream["primary_key"] + stream["join_keys"]:
            assert field_name in fields_by_stream[stream["file_name"]]
    for relationship in payload["relationships"]:
        assert (
            relationship["source_field"]
            in fields_by_stream[relationship["source_stream"]]
        )
        assert (
            relationship["target_field"]
            in fields_by_stream[relationship["target_stream"]]
        )


def test_archive_layout_cli_is_machine_readable(capsys) -> None:
    assert analysis_main(["schema", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == archive_layout()


def test_archive_validation_rejects_typed_schema_drift(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "result_records.jsonl", [{"result_id": [], "unexpected_field": True}]
    )

    report = validate_archive(tmp_path)

    assert not report["ok"]
    assert report["typed_record_counts"]["result_records.jsonl"] == 0
    assert any("schema_drift" in error for error in report["errors"])
    assert any(
        "field 'result_id' expected str; got list" in error
        for error in report["errors"]
    )
    assert any(
        "unknown fields: ['unexpected_field']" in error for error in report["errors"]
    )


def test_archive_validation_detects_linked_campaign_artifact_tampering(
    tmp_path: Path,
) -> None:
    campaign_input = tmp_path / "campaign_input.yaml"
    campaign_resolved = tmp_path / "campaign_resolved.json"
    campaign_input.write_text("name: publication-test\n")
    campaign_resolved.write_text('{"campaign": {"name": "publication-test"}}\n')

    input_sha = hashlib.sha256(campaign_input.read_bytes()).hexdigest()
    resolved_sha = hashlib.sha256(campaign_resolved.read_bytes()).hexdigest()
    prompt_roles = {
        role: {"system_prompt_sha256": character * 64}
        for role, character in (
            ("planner", "a"),
            ("supervisor", "b"),
            ("critic", "c"),
        )
    }
    (tmp_path / "run_provenance.json").write_text(
        json.dumps(
            {
                "schema_version": "v7.3.3_run_provenance_v2",
                "source": {"tree_sha256": "source"},
                "target": {"pdb_sha256": "target"},
                "model": {"content_sha256": "model"},
                "prompts": {"roles": prompt_roles},
                "campaign": {
                    "name": "publication-test",
                    "source_sha256": input_sha,
                    "input_artifact": str(campaign_input),
                    "input_artifact_sha256": input_sha,
                    "resolved_artifact": str(campaign_resolved),
                    "resolved_artifact_sha256": resolved_sha,
                },
            }
        )
    )

    assert validate_archive(tmp_path)["ok"]

    campaign_input.write_text("name: tampered\n")
    report = validate_archive(tmp_path)

    assert not report["ok"]
    assert any(
        "linked artifact SHA256 mismatch: campaign_input.yaml" in error
        for error in report["errors"]
    )


def test_analysis_cli_has_human_and_machine_readable_trace(
    tmp_path: Path,
    capsys,
) -> None:
    _write_jsonl(
        tmp_path / "evidence_summaries.jsonl",
        [
            {
                "tick_id": "v7r001",
                "target_id": "target",
                "state_label": "low_evidence",
                "strict_count": 0,
                "run_su_count": 0,
            }
        ],
    )

    assert (
        analysis_main(
            [
                "trace",
                "--archive-root",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert "v7r001: state=low_evidence" in capsys.readouterr().out

    assert (
        analysis_main(
            [
                "trace",
                "--archive-root",
                str(tmp_path),
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "trex.decision-trace.v1"
    assert payload["entries"][0]["schema_version"] == "trex.decision-trace-entry.v1"


def test_analysis_cli_rejects_missing_archive_without_creating_it(
    tmp_path: Path,
    capsys,
) -> None:
    missing = tmp_path / "missing"

    assert analysis_main(["summary", "--archive-root", str(missing)]) == 2

    assert "archive root is not a directory" in capsys.readouterr().err
    assert not missing.exists()


def test_archive_validation_classifies_lineage_and_rejects_broken_panel(
    tmp_path: Path,
) -> None:
    archive = Archive(tmp_path)
    archive.append(_archive_result("r0", []))
    archive.append(_archive_result("same", []))
    archive.append(_archive_result("child", ["candidate", "r0", "same", "external"]))
    feasibility = FeasibilityCheck(True, "runtime", True, True, True, True)
    for candidate_id in ("candidate", "same"):
        archive.append(
            ActionCandidate(
                candidate_id=candidate_id,
                hypothesis_ids=[],
                parent_result_id=None,
                method_family="test_backend",
                operator_id="test_operator",
                lane_id="test_lane",
                config_delta={},
                downstream_route_plan=[],
                estimated_cost_class="low",
                expected_signal="test",
                evidence_refs=[],
                feasibility=feasibility,
            )
        )
    archive.append(
        PanelSelection(
            panel_id="panel",
            K=1,
            selected_ids=["missing-result"],
            pareto_audit=[],
            diversity_bins={},
            hard_gate_failures=[],
            calibration_ref="test",
            panel_value=0.0,
        )
    )

    report = validate_archive(tmp_path)

    assert not report["ok"]
    assert report["lineage_references"] == {
        "total": 4,
        "candidate_only": 1,
        "result_only": 1,
        "ambiguous": 1,
        "external_or_unknown": 1,
        "external_or_unknown_examples": ["external"],
    }
    assert any(
        "panel_selections.jsonl references unknown result_id" in error
        for error in report["errors"]
    )


def test_archive_validation_rejects_duplicate_stable_event_ids(tmp_path: Path) -> None:
    row = {
        "launch_id": "duplicate-launch",
        "tick_id": "v7r001",
        "candidate_id": "candidate",
        "status": "launched",
        "resource_class_concrete": {"class": "low"},
        "why": "test",
    }
    _write_jsonl(tmp_path / "launch_decisions.jsonl", [row, row])

    report = validate_archive(tmp_path)

    assert not report["ok"]
    assert any("duplicate launch_id" in error for error in report["errors"])


def test_decision_trace_fails_closed_on_ambiguous_candidate_join(
    tmp_path: Path,
) -> None:
    _write_jsonl(
        tmp_path / "evidence_summaries.jsonl",
        [
            {
                "tick_id": "v7r001",
                "target_id": "target",
                "state_label": "low_evidence",
                "strict_count": 0,
                "run_su_count": 0,
            }
        ],
    )
    _write_jsonl(
        tmp_path / "action_candidates.jsonl",
        [
            {"candidate_id": "same", "method_family": "first"},
            {"candidate_id": "same", "method_family": "second"},
        ],
    )
    _write_jsonl(
        tmp_path / "launch_decisions.jsonl",
        [{"tick_id": "v7r001", "candidate_id": "same", "status": "launched"}],
    )

    trace = decision_trace(tmp_path, limit=1)
    joined = trace[0]["launches"][0]

    assert joined["candidate_join_status"] == "ambiguous"
    assert joined["candidate_match_count"] == 2
    assert joined["family"] is None


def test_chain_identity_maintenance_hashes_reject_unreviewed_changes(tmp_path):
    import shutil
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("trex_parity_verifier", root / "scripts/verify_behavioral_parity.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in module.MAINTENANCE_MODULE_SHA256:
        shutil.copy2(root / "trex" / name, tmp_path / name)
    assert all(row["accepted"] for row in module._maintenance_module_checks(tmp_path))
    helper = tmp_path / "af2_chain_identity.py"
    helper.write_text(helper.read_text() + "\nUNREVIEWED_BEHAVIOR = True\n")
    checks = module._maintenance_module_checks(tmp_path)
    assert not next(row["accepted"] for row in checks if row["module"] == helper.name)
