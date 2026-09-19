"""Shared deduplication trust gates.

These helpers decide whether advisory dedup-derived counts are safe to use in
deterministic control. Keep them small and dependency-free so Planner,
EvidenceReducer, Selector, and CandidateBuilder cannot drift apart.
"""

from __future__ import annotations

from typing import Any


def near_miss_dedup_trusted(
    status: str | None,
    coverage: float | None,
    *,
    near_miss_count: int | None = None,
) -> bool:
    """Whether near-miss structural counts are safe for allocation decisions.

    Near-miss records are qualitative rescue evidence, not official SU credit,
    but their counts still steer state classification and Selector floors. If
    near-miss-only Foldseek dedup degraded, raw result_id fallbacks can turn one
    repeated basin into many apparent rescue opportunities. Keep raw exemplars
    visible elsewhere, but do not let an untrusted count drive deterministic
    control.
    """

    try:
        raw_n = int(near_miss_count or 0)
    except (TypeError, ValueError):
        raw_n = 0
    st = str(status or "legacy_or_unknown")
    if raw_n <= 0 or st == "no_near_miss":
        return True
    if st == "disabled":
        return False
    if st in {"failed", "no_binary", "no_structures"}:
        return False
    if st in {"ok", "cached_ok"}:
        if coverage is None:
            return True
        try:
            return float(coverage) >= 0.999
        except (TypeError, ValueError):
            return False
    # Legacy/offline summaries may not have near-miss provenance. Preserve
    # backwards-compatible interpretation for hand-authored tests and old JSONL,
    # but live production statuses above are explicit and therefore enforced.
    return st == "legacy_or_unknown"


def near_miss_dedup_trusted_from_evidence(evidence: Any) -> bool:
    """EvidenceSummary-shaped wrapper for near_miss_dedup_trusted."""

    raw_n = getattr(evidence, "near_miss_count", 0)
    # EvidenceReducer keeps raw near-miss exemplars/provenance visible to the
    # LLM even when deterministic near-miss counts are zeroed because dedup is
    # degraded. Deterministic control must still treat those exemplars as
    # untrusted rescue counts.
    if not raw_n and getattr(evidence, "production_near_miss_ids", None):
        raw_n = len(getattr(evidence, "production_near_miss_ids", None) or [])
    return near_miss_dedup_trusted(
        getattr(evidence, "near_miss_dedup_status", "legacy_or_unknown"),
        getattr(evidence, "near_miss_dedup_coverage", None),
        near_miss_count=raw_n,
    )
