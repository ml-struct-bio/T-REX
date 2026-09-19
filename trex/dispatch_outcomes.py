"""Operator-facing interpretation of immutable dispatch audit records.

DispatchRecord.status is intentionally a small, backward-compatible wire
vocabulary. Some dispatch_failed rows describe work that never attempted a
backend launch, such as a capacity deferral or a stale queued action. This
module preserves the raw status while exposing a clearer derived outcome for
summaries and decision traces.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any


def _record_value(record: object, field_name: str) -> Any:
    if isinstance(record, Mapping):
        return record.get(field_name)
    return getattr(record, field_name, None)


def classify_dispatch_outcome(record: object) -> str:
    """Return an operator-facing outcome without changing the archived row."""

    status = str(_record_value(record, "status") or "<missing>")
    if status != "dispatch_failed":
        return status

    dispatch_id = str(_record_value(record, "dispatch_id") or "")
    why = str(_record_value(record, "why") or "")
    if dispatch_id.startswith("dispatch_deferred_") or why.startswith(
        "high_cost_inflight_cap"
    ):
        return "capacity_deferred"
    if (
        dispatch_id.startswith(("stale_prefetch_", "deep_stall_throttle_"))
        or why.startswith("stale scientific prefetch")
        or "cancelled deterministic diagnostic score-conversion reserve before dispatch"
        in why
    ):
        return "cancelled_before_start"
    return status


def count_dispatch_outcomes(records: Iterable[object]) -> dict[str, int]:
    """Count derived outcomes using deterministic key order."""

    counts = Counter(classify_dispatch_outcome(record) for record in records)
    return dict(sorted(counts.items()))


__all__ = ["classify_dispatch_outcome", "count_dispatch_outcomes"]
