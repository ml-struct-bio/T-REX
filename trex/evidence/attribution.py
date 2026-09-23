"""Lineage and canonical-SU attribution primitives.

These functions define who receives scientific success and compute-cost credit.
They are kept independent from evidence aggregation so reducers and controllers
share one deterministic attribution contract.
"""

from __future__ import annotations

from ..refilter_roles import (
    CANONICAL_SCORE_CONVERSION,
    PARENT_MODEL_REFOLD,
    infer_refilter_role,
)
from ..schemas import ActionCandidate, ResultRecord
from ..success_criteria import is_strict_success


def _su_key(r: ResultRecord) -> str | None:
    """Official SU dedup key: strict-only Foldseek cluster id.

    No fallback is allowed for SU credit. Falling back to refilter_source or
    result_id makes raw/offline reducer replays over-count strict records as
    structurally unique successes when Foldseek bins are absent or degraded.
    Strict records without this key still contribute to strict_count/quality
    diagnostics, but not to new_su or SU/GPU-h evidence.
    """
    key = (r.bins or {}).get("foldseek_su")
    return str(key) if key else None


def _infer_refilter_role_for_record(
    r: ResultRecord,
    *,
    spawning_actions: dict[str, ActionCandidate] | None = None,
) -> str | None:
    """Role of a refilter-like ResultRecord, with legacy archive inference."""
    if r.backend_family != "structure_refilter":
        return None
    b = r.bins or {}
    if b.get("refilter_role"):
        return b["refilter_role"]
    ac = (spawning_actions or {}).get(r.result_id)
    if ac is not None:
        role = infer_refilter_role(ac)
        if role:
            return role
    if r.backend_family == "structure_refilter":
        if any((pid or "").startswith("chain_") for pid in (r.parent_ids or [])):
            return CANONICAL_SCORE_CONVERSION
        return PARENT_MODEL_REFOLD


def resolve_generating_record(
    r: ResultRecord,
    *,
    by_result_id: dict[str, ResultRecord],
    spawning_actions: dict[str, ActionCandidate],
    _depth: int = 0,
) -> ResultRecord:
    """The RECORD that should receive primary SU/strategy credit.

    Only canonical AF2 score-conversion records are accounting plumbing: those
    walk through ``bins.refilter_source`` / ``parent_ids[1]`` / the spawning
    action's parent to credit the upstream generator or redesign. Intentional
    parent_model_refold records are adaptive refold actions, so they stay on the
    structure_refilter family/role instead of being collapsed into their parent.
    """
    if r.backend_family != "structure_refilter" or _depth > 8:
        return r
    if _infer_refilter_role_for_record(
        r, spawning_actions=spawning_actions
    ) != CANONICAL_SCORE_CONVERSION:
        return r
    b = r.bins or {}
    # Try each parent reference until one resolves to a result. A file path in
    # refilter_source must not hide a valid parent_ids reference.
    ac = spawning_actions.get(r.result_id)
    sources = [
        b.get("refilter_source"),
        r.parent_ids[1] if len(r.parent_ids or []) >= 2 else None,
        ac.parent_result_id if ac is not None else None,
    ]
    parent = next(
        (by_result_id[s] for s in sources if s and s in by_result_id), None
    )
    if parent is None:
        return r  # unresolvable source → keep refilter (rare)
    return resolve_generating_record(
        parent, by_result_id=by_result_id,
        spawning_actions=spawning_actions, _depth=_depth + 1,
    )


def resolve_generating_family(
    r: ResultRecord,
    *,
    by_result_id: dict[str, ResultRecord],
    spawning_actions: dict[str, ActionCandidate],
) -> str:
    """Generating family (the backend_family of resolve_generating_record)."""
    return resolve_generating_record(
        r, by_result_id=by_result_id, spawning_actions=spawning_actions,
    ).backend_family


def _refilter_role_for_record(
    r: ResultRecord,
    *,
    spawning_actions: dict[str, ActionCandidate] | None = None,
) -> str | None:
    return _infer_refilter_role_for_record(r, spawning_actions=spawning_actions)


def is_canonical_su_record(
    r: ResultRecord,
    spawning_actions: dict[str, ActionCandidate] | None = None,
) -> bool:
    """Strict record eligible for official SU accounting.

    Official T-REX SU uses the canonical single-model AF2 score-conversion gate.
    Current parsers keep diagnostic-native generator scores out of strict keys;
    this helper therefore only blocks refilter rows whose role is explicitly
    advisory (parent-model/cross-model refolds). Direct strict rows remain
    backward-compatible for migrated archives and synthetic tests.
    """
    if r.exit_status != "ok" or not is_strict_success(r.metrics):
        return False
    explicit_role = (r.bins or {}).get("refilter_role")
    if explicit_role is None:
        ac = (spawning_actions or {}).get(r.result_id)
        explicit_role = infer_refilter_role(ac) if ac is not None else None
    if r.backend_family == "structure_refilter":
        # Unannotated legacy structure_refilter rows are treated as canonical
        # score-conversion for backward-compatible replay. Explicit
        # parent_model_refold rows are advisory and cannot mint SU.
        return explicit_role != PARENT_MODEL_REFOLD
    return True


# Compatibility alias for downstream callers.
_is_canonical_su_record = is_canonical_su_record


def _direct_parent_record(
    r: ResultRecord,
    *,
    by_result_id: dict[str, ResultRecord],
    spawning_actions: dict[str, ActionCandidate],
) -> ResultRecord | None:
    """Resolve the concrete parent result. Prefer the spawning candidate parent over mixed
    candidate/result references in ResultRecord.parent_ids.
    """
    ac = spawning_actions.get(r.result_id)
    candidates: list[str | None] = [
        ac.parent_result_id if ac is not None else None,
    ]
    if len(r.parent_ids or []) >= 2:
        candidates.append(r.parent_ids[1])
    if r.parent_ids:
        candidates.append(r.parent_ids[0])
    for parent_id in candidates:
        if parent_id and parent_id in by_result_id:
            return by_result_id[parent_id]
    return None


def _lineage_parent_records(
    r: ResultRecord,
    *,
    by_result_id: dict[str, ResultRecord],
    spawning_actions: dict[str, ActionCandidate],
    max_depth: int = 8,
) -> list[ResultRecord]:
    """Ordered concrete ancestors needed to reproduce a parent-bound route."""
    out: list[ResultRecord] = []
    seen = {r.result_id}
    cur = _direct_parent_record(
        r, by_result_id=by_result_id, spawning_actions=spawning_actions,
    )
    depth = 0
    while cur is not None and cur.result_id not in seen and depth < max_depth:
        out.append(cur)
        seen.add(cur.result_id)
        depth += 1
        cur = _direct_parent_record(
            cur, by_result_id=by_result_id, spawning_actions=spawning_actions,
        )
    return out


def _score_conversion_parent_record(
    r: ResultRecord,
    *,
    by_result_id: dict[str, ResultRecord],
    spawning_actions: dict[str, ActionCandidate],
) -> ResultRecord | None:
    """Immediate generator/redesign record scored by a canonical AF2 conversion."""
    ac = spawning_actions.get(r.result_id)
    candidates = [ac.parent_result_id if ac is not None else None]
    bins = r.bins or {}
    candidates.extend([
        bins.get("refilter_source"),
        r.parent_ids[1] if len(r.parent_ids or []) >= 2 else None,
    ])
    for parent_id in candidates:
        if parent_id and parent_id in by_result_id:
            return by_result_id[parent_id]
    return None


def _score_conversion_lineage_records(
    r: ResultRecord,
    *,
    by_result_id: dict[str, ResultRecord],
    spawning_actions: dict[str, ActionCandidate],
) -> list[ResultRecord]:
    """Immediate scored record plus its concrete parent ancestors."""
    parent = _score_conversion_parent_record(
        r, by_result_id=by_result_id, spawning_actions=spawning_actions,
    )
    if parent is None:
        return []
    return [
        parent,
        *_lineage_parent_records(
            parent,
            by_result_id=by_result_id,
            spawning_actions=spawning_actions,
        ),
    ]
