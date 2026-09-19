from __future__ import annotations

import json
from pathlib import Path

from trex.analysis import decision_trace


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_trace_matches_production_tick_ids_and_deduplicates_updates(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "evidence_summaries.jsonl", [
        {"tick_id": "v7r007", "target_id": "T"},
        {"tick_id": "v7r008", "target_id": "T"},
    ])
    card = {
        "tick_created": 7,
        "hypothesis_id": "hyp_0007_00",
        "claim": "first revision",
    }
    revised = {**card, "claim": "latest revision"}
    _write(tmp_path / "hypothesis_cards.jsonl", [card, revised])
    _write(tmp_path / "launch_decisions.jsonl", [{
        "tick_id": "v7r008",
        "candidate_id": "chain_only",
        "status": "launched",
    }])

    trace = decision_trace(tmp_path, limit=1)

    assert trace[0]["tick_id"] == "v7r007"
    assert len(trace[0]["hypotheses"]) == 1
    assert trace[0]["hypotheses"][0]["claim"] == "latest revision"
