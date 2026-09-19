"""Presentation-only contracts for existing, synthetic archive records."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trex.analysis import campaign_summary, main


@pytest.fixture
def record_archive(tmp_path: Path) -> Path:
    rows = [
        {
            "role": "planner", "model": "vllm/test-model",
            "tokens_in": 40, "tokens_out": 11, "latency_s": 1.5,
            "parse_status": "ok", "fallback_triggered": False,
        },
        {
            "role": "supervisor", "model": "vllm/test-model",
            "tokens_in": 0, "tokens_out": 0, "latency_s": 0.0,
            "parse_status": "no_hypotheses_or_candidates",
            "fallback_triggered": True,
        },
        {
            "role": "critic", "model": "deterministic_guard",
            "tokens_in": 0, "tokens_out": 0, "latency_s": 0.0,
            "parse_status": "ok", "fallback_triggered": False,
        },
    ]
    (tmp_path / "llm_call_records.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return tmp_path


def test_summary_preserves_legacy_json_record_counts(record_archive, capsys) -> None:
    assert main(["summary", "--archive-root", str(record_archive), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["schema_version"] == "trex.analysis-summary.v1"
    assert payload == campaign_summary(record_archive)
    assert payload["llm_usage"] == {
        "calls": 3, "tokens_in": 40, "tokens_out": 11,
        "by_role": {
            "planner": {
                "calls": 1, "tokens_in": 40, "tokens_out": 11,
                "latency_s": 1.5, "parse_status": {"ok": 1}, "fallback_calls": 0,
            },
            "supervisor": {
                "calls": 1, "tokens_in": 0, "tokens_out": 0,
                "latency_s": 0.0,
                "parse_status": {"no_hypotheses_or_candidates": 1},
                "fallback_calls": 1,
            },
            "critic": {
                "calls": 1, "tokens_in": 0, "tokens_out": 0,
                "latency_s": 0.0, "parse_status": {"ok": 1}, "fallback_calls": 0,
            },
        },
    }


def test_text_summary_does_not_label_records_as_api_calls(record_archive, capsys) -> None:
    assert main(["summary", "--archive-root", str(record_archive)]) == 0
    output = capsys.readouterr().out

    assert "LLM records: count=3 input_tokens=40 output_tokens=11" in output
    assert "not API invocation counts" in output
    assert "calls=3" not in output


@pytest.mark.parametrize("as_json", [False, True])
def test_summary_leaves_archive_bytes_and_paths_unchanged(
    record_archive, capsys, as_json,
) -> None:
    extra = record_archive / "untouched.txt"
    extra.write_bytes(b"Existing user artifact.\n")
    before = {path.name: path.read_bytes() for path in record_archive.iterdir()}
    args = ["summary", "--archive-root", str(record_archive)]
    if as_json:
        args.append("--json")

    assert main(args) == 0
    capsys.readouterr()

    assert {path.name: path.read_bytes() for path in record_archive.iterdir()} == before


def test_missing_evidence_is_not_reported_as_a_measured_zero(tmp_path) -> None:
    summary = campaign_summary(tmp_path)

    assert summary["latest_tick"] is None
    assert summary["endpoint"]["structure_unique_successes_tm_live"] is None
    assert summary["endpoint"]["su_per_worker_wall_gpu_h"] is None
    assert summary["budget"]["worker_wall_gpu_h"] is None
    assert summary["llm_usage"]["calls"] == 0


@pytest.mark.parametrize("parse_status", ["ok", "call_error", "timeout"])
def test_zero_recorded_tokens_do_not_remove_a_record(tmp_path, parse_status) -> None:
    (tmp_path / "llm_call_records.jsonl").write_text(
        json.dumps({"role": "planner", "parse_status": parse_status}) + "\n",
        encoding="utf-8",
    )

    usage = campaign_summary(tmp_path)["llm_usage"]

    assert usage["calls"] == 1
    assert usage["tokens_in"] == usage["tokens_out"] == 0
    assert usage["by_role"]["planner"]["parse_status"] == {parse_status: 1}
