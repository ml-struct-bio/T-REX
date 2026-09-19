"""Archive append/read smoke tests."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from trex.archive import Archive
from trex.schemas import (
    HypothesisCard,
    LaunchDecision,
    PredictedChange,
    PreserveConstraint,
    ResultRecord,
    RuntimeBucket,
)


def _result(rid: str, target: str = "t") -> ResultRecord:
    return ResultRecord(
        result_id=rid,
        parent_ids=[],
        target_id=target,
        backend_family="complexa_beam",
        runtime_bucket_id="rb1",
        metrics={"pLDDT": 85.0, "iPAE": 0.3},
        metrics_calibrated={},
        route_lineage=[],
        gpu_h=1.0,
        exit_status="ok",
    )


def _hyp(hid: str, tick: int, *, status: str = "active", target: str = "t") -> HypothesisCard:
    return HypothesisCard(
        hypothesis_id=hid,
        target_id=target,
        tick_created=tick,
        claim="x",
        mode_affinity={"exploit": 0.1, "rescue": 0.8, "explore": 0.1},
        evidence_refs=["e1"],
        predicted_metric_changes=[PredictedChange("iPAE", "decrease", ["b"], 0.20, None)],
        preserve_constraints=[PreserveConstraint("pLDDT", 0.05)],
        recommended_action_families=["proteinmpnn_redesign"],
        status=status,  # type: ignore[arg-type]
    )


def test_append_and_iter(tmp_path: Path):
    arc = Archive(tmp_path / "a1")
    arc.append(_result("r1"))
    arc.append(_result("r2"))
    arc.append(_result("r3"))

    ids = [r.result_id for r in arc.iter_records(ResultRecord)]
    assert ids == ["r1", "r2", "r3"]
    assert arc.count(ResultRecord) == 3


def test_iter_records_cache_invalidates_on_local_and_external_append(tmp_path: Path):
    root = tmp_path / "cache_invalidate"
    arc1 = Archive(root)
    arc2 = Archive(root)
    arc1.append(_result("r1"))
    assert [r.result_id for r in arc1.iter_records(ResultRecord)] == ["r1"]
    # Cache hit should return the same content before the file changes.
    assert [r.result_id for r in arc1.iter_records(ResultRecord)] == ["r1"]

    arc2.append(_result("r2"))
    assert [r.result_id for r in arc1.iter_records(ResultRecord)] == ["r1", "r2"]

    arc1.append(_result("r3"))
    assert [r.result_id for r in arc1.iter_records(ResultRecord)] == ["r1", "r2", "r3"]


def test_iter_records_cache_is_mutation_isolated(tmp_path: Path):
    arc = Archive(tmp_path / "cache_mutation")
    arc.append(_result("r1"))

    first = next(arc.iter_records(ResultRecord))
    first.bins["foldseek_su"] = "transient_cluster"
    first.metrics["pLDDT"] = 1.0

    second = next(arc.iter_records(ResultRecord))
    assert "foldseek_su" not in second.bins
    assert second.metrics["pLDDT"] == 85.0


def test_duplicate_result_id_is_rejected(tmp_path: Path):
    arc = Archive(tmp_path / "dups")
    arc.append(_result("r1"))

    with pytest.raises(ValueError, match="duplicate ResultRecord.result_id"):
        arc.append(_result("r1"))


def test_iter_skips_partial_lines(tmp_path: Path):
    arc = Archive(tmp_path / "a2")
    arc.append(_result("r1"))
    # corrupt the file
    p = tmp_path / "a2" / "result_records.jsonl"
    p.write_text(p.read_text() + "not-json\n")
    arc.append(_result("r2"))
    ids = [r.result_id for r in arc.iter_records(ResultRecord)]
    assert ids == ["r1", "r2"]


def test_retrieve_active_excludes_retired(tmp_path: Path):
    arc = Archive(tmp_path / "a3")
    arc.append(_hyp("h1", 1, status="active"))
    arc.append(_hyp("h2", 2, status="retired"))
    arc.append(_hyp("h3", 3, status="supported"))
    active = arc.retrieve_active_hypotheses()
    ids = [h.hypothesis_id for h in active]
    assert "h2" not in ids
    assert ids == ["h1", "h3"]  # active work precedes terminal memory


def test_active_hypothesis_not_hidden_by_terminal_memory_cap(tmp_path: Path):
    arc = Archive(tmp_path / "active_not_hidden")
    arc.append(_hyp("old_active", 0, status="active"))
    for tick in range(1, 10):
        arc.append(_hyp(f"terminal_{tick}", tick, status="supported"))

    prompt_memory = arc.retrieve_active_hypotheses(limit=8)
    lifecycle_memory = arc.retrieve_active_hypotheses(
        limit=None, include_terminal=False,
    )

    assert prompt_memory[0].hypothesis_id == "old_active"
    assert len(prompt_memory) == 8
    assert [h.hypothesis_id for h in lifecycle_memory] == ["old_active"]


def test_target_filter(tmp_path: Path):
    arc = Archive(tmp_path / "a4")
    arc.append(_hyp("h1", 1, target="a"))
    arc.append(_hyp("h2", 2, target="b"))
    a = arc.retrieve_active_hypotheses(target_id="a")
    assert [h.hypothesis_id for h in a] == ["h1"]


def test_summary_reflects_writes(tmp_path: Path):
    arc = Archive(tmp_path / "a5")
    arc.append(_result("r1"))
    arc.append(
        RuntimeBucket(
            bucket_id="rb1",
            container_digest=None,
            ckpt_digests={},
            scoring_script_digest=None,
            calibration_version="v1",
            created_at="2026-05-24T00:00:00Z",
        )
    )
    arc.append(
        LaunchDecision(
            launch_id="L1",
            tick_id="t1",
            candidate_id="c1",
            status="launched",
            resource_class_concrete={"class": "low", "source": "test"},
            why="...",
        )
    )
    s = arc.summary()
    assert s["result_records.jsonl"] == 1
    assert s["runtime_buckets.jsonl"] == 1
    assert s["launch_decisions.jsonl"] == 1


def test_unknown_record_type_raises(tmp_path: Path):
    arc = Archive(tmp_path / "a6")
    with pytest.raises(TypeError):
        arc.append({"not": "a dataclass"})


def test_append_many(tmp_path: Path):
    arc = Archive(tmp_path / "a7")
    arc.append_many([_result(f"r{i}") for i in range(5)])
    assert arc.count(ResultRecord) == 5



def test_iter_records_counts_schema_drift_skips(tmp_path: Path):
    arc = Archive(tmp_path / "schema_drift")
    arc.append(_result("r1"))
    p = tmp_path / "schema_drift" / "result_records.jsonl"
    p.write_text(p.read_text() + "{}\nnot-json\n")

    ids = [r.result_id for r in arc.iter_records(ResultRecord)]
    assert ids == ["r1"]
    counts = arc.read_skip_counts()
    assert counts["result_records.jsonl:schema_drift"] == 1
    assert counts["result_records.jsonl:json_decode"] == 1


def test_duplicate_result_id_recheck_across_archive_instances(tmp_path: Path):
    root = tmp_path / "dups_multi_instance"
    arc1 = Archive(root)
    arc2 = Archive(root)
    arc1.append(_result("r1"))
    arc2.append(_result("r2"))

    with pytest.raises(ValueError, match="duplicate ResultRecord.result_id"):
        arc1.append(_result("r2"))


def test_result_id_refresh_reads_only_new_archive_tail(tmp_path: Path):
    root = tmp_path / "incremental_ids"
    arc1 = Archive(root)
    arc2 = Archive(root)
    arc1.append(_result("r1"))
    arc1.append(_result("r2"))
    arc2.append(_result("r3"))

    with patch(
        "trex.archive.json.loads", wraps=json.loads,
    ) as loads:
        arc1.append(_result("r4"))

    # arc1 already indexed r1/r2; only arc2's newly appended r3 is parsed.
    assert loads.call_count == 1
    assert [r.result_id for r in arc1.iter_records(ResultRecord)] == [
        "r1", "r2", "r3", "r4",
    ]


def test_failed_result_append_does_not_poison_in_memory_id_set(tmp_path: Path):
    arc = Archive(tmp_path / "append_failure")
    real_append = arc._append_line
    with patch.object(arc, "_append_line", side_effect=OSError("disk error")):
        with pytest.raises(OSError, match="disk error"):
            arc.append(_result("retryable"))

    arc._append_line = real_append
    arc.append(_result("retryable"))
    assert [r.result_id for r in arc.iter_records(ResultRecord)] == ["retryable"]


def test_retrieve_active_dedupes_to_latest_lifecycle_row(tmp_path: Path):
    arc = Archive(tmp_path / "hyp_dedupe")
    original = _hyp("h1", 1, status="active")
    updated = _hyp("h1", 1, status="supported")
    object.__setattr__(updated, "support_points", 2)
    arc.append(original)
    arc.append(_hyp("h2", 2, status="active"))
    arc.append(updated)

    active = arc.retrieve_active_hypotheses()
    h1_rows = [h for h in active if h.hypothesis_id == "h1"]
    assert len(h1_rows) == 1
    assert h1_rows[0].status == "supported"
    assert h1_rows[0].support_points == 2
    assert [h.hypothesis_id for h in active[:2]] == ["h2", "h1"]
