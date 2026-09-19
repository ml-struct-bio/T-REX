"""Deterministic candidate ranking, diversity, and deferred backfill."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..schemas import ActionCandidate


MODES = ("exploit", "rescue", "explore")


@dataclass(frozen=True)
class BatchDiversityPolicy:
    """Resolved batch-diversity limits and duplicate-pressure penalties."""

    max_per_family: int
    overrepresented_families: frozenset[str]


@dataclass(frozen=True)
class RankingCandidateContext:
    """One candidate plus the policy facts required by final ranking."""

    candidate: ActionCandidate
    mode: str | None
    priority: int
    rank_in_mode: int | None
    global_rank: int | None


@dataclass(frozen=True)
class SelectionRankingRequest:
    """Explicit inputs to quota-constrained candidate ranking."""

    eligible_candidates: Sequence[RankingCandidateContext]
    cost_deferred_candidates: Sequence[RankingCandidateContext]
    route_deferred_candidates: Sequence[RankingCandidateContext]
    cross_family_escape_candidates: Sequence[RankingCandidateContext]
    quotas: Mapping[str, int]
    available_slots: int
    diversity_policy: BatchDiversityPolicy
    high_cost_batch_hard_caps: Mapping[str, int]
    has_mode_information: bool
    global_priority_used: bool
    top_n_rejected_per_mode: int
    cross_family_escape_floor_enabled: bool


@dataclass(frozen=True)
class SelectionRankingResult:
    """Selected candidates, realized modes, rejection pools, and audit data."""

    selected_candidates: tuple[ActionCandidate, ...]
    selection_modes: dict[str, str]
    rejected_per_mode: dict[str, tuple[ActionCandidate, ...]]
    global_priority_backfill_ids: tuple[str, ...]
    forced_cross_family_escape_floor: dict[str, Any] | None
    audit_log: tuple[str, ...]


@dataclass
class _SelectionState:
    remaining: list[RankingCandidateContext]
    selected: list[RankingCandidateContext] = field(default_factory=list)
    selection_modes: dict[str, str] = field(default_factory=dict)
    rejected_per_mode: dict[str, list[ActionCandidate]] = field(
        default_factory=lambda: {mode: [] for mode in MODES}
    )
    global_priority_backfill_ids: list[str] = field(default_factory=list)

    def family_counts(self) -> dict[str, int]:
        families = {
            context.candidate.method_family
            for context in self.selected
        }
        return {
            family: sum(
                1
                for context in self.selected
                if context.candidate.method_family == family
            )
            for family in families
        }

    def add(self, context: RankingCandidateContext, mode: str) -> None:
        self.selected.append(context)
        self.selection_modes[context.candidate.candidate_id] = mode

    def remove_from_remaining(
        self,
        contexts: Sequence[RankingCandidateContext],
    ) -> None:
        selected_ids = {
            context.candidate.candidate_id
            for context in contexts
        }
        self.remaining = [
            context
            for context in self.remaining
            if context.candidate.candidate_id not in selected_ids
        ]


def resolve_batch_diversity_policy(
    *,
    state_label: str,
    category_b_enabled: bool,
    available_slots: int,
    default_max_per_family: int,
    duplicate_fraction: float | None,
    recent_started_families: Sequence[str] | None,
    duplicate_pressure_fraction: float,
    family_share_cap: float,
    family_window_k: int,
) -> BatchDiversityPolicy:
    """Resolve batch caps and recent duplicate-family pressure."""

    if (
        not category_b_enabled
        and state_label in ("productive", "low_evidence")
    ):
        max_per_family = available_slots
    else:
        max_per_family = default_max_per_family

    overrepresented_families: set[str] = set()
    if (
        state_label in ("productive_duplicate", "strict_duplicate_collapse")
        and recent_started_families
        and (
            state_label == "strict_duplicate_collapse"
            or (
                duplicate_fraction is not None
                and duplicate_fraction >= duplicate_pressure_fraction
            )
        )
    ):
        family_window = list(recent_started_families)[-family_window_k:]
        denominator = max(1, len(family_window))
        for family in set(family_window):
            if family_window.count(family) / denominator >= family_share_cap:
                overrepresented_families.add(family)

    return BatchDiversityPolicy(
        max_per_family=max_per_family,
        overrepresented_families=frozenset(overrepresented_families),
    )


def _enforce_batch_diversity(
    sorted_candidates: list[ActionCandidate],
    n: int,
    max_per_family: int,
    *,
    hard_max_by_family: dict[str, int] | None = None,
    initial_family_counts: dict[str, int] | None = None,
) -> list[ActionCandidate]:
    """Select up to ``n`` candidates under soft and hard family caps."""

    selected: list[ActionCandidate] = []
    family_counts: dict[str, int] = dict(initial_family_counts or {})
    hard_max_by_family = hard_max_by_family or {}

    def has_hard_cap_room(candidate: ActionCandidate) -> bool:
        family = candidate.method_family
        if family not in hard_max_by_family:
            return True
        return family_counts.get(family, 0) < max(
            0,
            int(hard_max_by_family[family]),
        )

    for candidate in sorted_candidates:
        if len(selected) >= n:
            break
        family = candidate.method_family
        if not has_hard_cap_room(candidate):
            continue
        if family_counts.get(family, 0) >= max_per_family:
            continue
        selected.append(candidate)
        family_counts[family] = family_counts.get(family, 0) + 1

    if len(selected) < n:
        for candidate in sorted_candidates:
            if len(selected) >= n:
                break
            if candidate in selected or not has_hard_cap_room(candidate):
                continue
            selected.append(candidate)
            family = candidate.method_family
            family_counts[family] = family_counts.get(family, 0) + 1
    return selected


def _contexts_for_candidates(
    contexts: Sequence[RankingCandidateContext],
    candidates: Sequence[ActionCandidate],
) -> list[RankingCandidateContext]:
    context_by_id = {
        context.candidate.candidate_id: context
        for context in contexts
    }
    return [
        context_by_id[candidate.candidate_id]
        for candidate in candidates
    ]


def _select_with_diversity(
    request: SelectionRankingRequest,
    state: _SelectionState,
    candidates: Sequence[RankingCandidateContext],
    n_slots: int,
) -> list[RankingCandidateContext]:
    ordered = sorted(candidates, key=lambda context: context.priority)
    selected_candidates = _enforce_batch_diversity(
        [context.candidate for context in ordered],
        n_slots,
        request.diversity_policy.max_per_family,
        hard_max_by_family=dict(request.high_cost_batch_hard_caps),
        initial_family_counts=state.family_counts(),
    )
    return _contexts_for_candidates(ordered, selected_candidates)


def _select_global_priority(
    request: SelectionRankingRequest,
    state: _SelectionState,
) -> None:
    globally_ranked = sorted(
        (
            context
            for context in state.remaining
            if context.global_rank is not None
        ),
        key=lambda context: (
            int(context.global_rank or 10**6),
            context.candidate.candidate_id,
        ),
    )
    selected_candidates = _enforce_batch_diversity(
        [context.candidate for context in globally_ranked],
        request.available_slots,
        request.available_slots,
        hard_max_by_family=dict(request.high_cost_batch_hard_caps),
    )
    selected_contexts = _contexts_for_candidates(
        globally_ranked,
        selected_candidates,
    )
    selected_ids = {
        context.candidate.candidate_id
        for context in selected_contexts
    }
    for context in selected_contexts:
        state.add(context, context.mode or "exploit")
    state.remove_from_remaining(selected_contexts)

    for mode in MODES:
        state.rejected_per_mode[mode] = [
            context.candidate
            for context in globally_ranked
            if context.candidate.candidate_id not in selected_ids
            and context.mode == mode
        ][: request.top_n_rejected_per_mode]

    n_unfilled = request.available_slots - len(state.selected)
    if n_unfilled <= 0 or not state.remaining:
        return
    backfill = _select_with_diversity(
        request,
        state,
        state.remaining,
        n_unfilled,
    )
    for context in backfill:
        state.add(context, context.mode or "exploit")
        state.global_priority_backfill_ids.append(
            context.candidate.candidate_id
        )
    state.remove_from_remaining(backfill)


def _select_mode_quotas(
    request: SelectionRankingRequest,
    state: _SelectionState,
) -> None:
    for mode in MODES:
        if request.global_priority_used:
            return
        mode_quota = request.quotas.get(mode, 0)
        if mode_quota <= 0:
            continue
        mode_candidates = (
            [
                context
                for context in state.remaining
                if context.mode == mode
            ]
            if request.has_mode_information
            else list(state.remaining)
        )
        ordered = sorted(
            mode_candidates,
            key=lambda context: context.priority,
        )
        selected = _select_with_diversity(
            request,
            state,
            ordered,
            mode_quota,
        )
        selected_ids = {
            context.candidate.candidate_id
            for context in selected
        }
        for context in selected:
            state.add(context, mode)
        state.rejected_per_mode[mode] = [
            context.candidate
            for context in ordered
            if context.candidate.candidate_id not in selected_ids
        ][: request.top_n_rejected_per_mode]
        state.remove_from_remaining(selected)


def _backfill_from(
    request: SelectionRankingRequest,
    state: _SelectionState,
    candidates: Sequence[RankingCandidateContext],
) -> None:
    n_unfilled = request.available_slots - len(state.selected)
    if request.global_priority_used or n_unfilled <= 0 or not candidates:
        return
    selected = _select_with_diversity(
        request,
        state,
        candidates,
        n_unfilled,
    )
    for context in selected:
        state.add(context, context.mode or "exploit")


def _apply_cross_family_escape_floor(
    request: SelectionRankingRequest,
    state: _SelectionState,
    context_by_id: Mapping[str, RankingCandidateContext],
) -> tuple[dict[str, Any] | None, tuple[str, ...]]:
    if (
        not request.cross_family_escape_floor_enabled
        or request.global_priority_used
        or request.available_slots <= 0
    ):
        return None, ()
    cross_family_candidates = sorted(
        request.cross_family_escape_candidates,
        key=lambda context: context.priority,
    )
    if not cross_family_candidates:
        return None, ()

    selected_cross_ids = {
        context.candidate.candidate_id
        for context in state.selected
        if context.candidate.candidate_id.startswith(
            "evidence_fallback_cross_family"
        )
    }
    target_cross = min(
        1,
        len(cross_family_candidates),
        request.available_slots,
    )
    added: list[str] = []
    replaced: list[dict[str, str]] = []
    audit_log: list[str] = []
    for context in cross_family_candidates:
        candidate_id = context.candidate.candidate_id
        if len(selected_cross_ids) >= target_cross:
            break
        if candidate_id in selected_cross_ids:
            continue
        if len(state.selected) < request.available_slots:
            state.add(context, context.mode or "explore")
            selected_cross_ids.add(candidate_id)
            added.append(candidate_id)
            continue

        victims = [
            selected_context
            for selected_context in state.selected
            if not selected_context.candidate.candidate_id.startswith(
                "evidence_fallback_cross_family"
            )
            and (
                context_by_id[
                    selected_context.candidate.candidate_id
                ].rank_in_mode is None
                or int(context_by_id[
                    selected_context.candidate.candidate_id
                ].rank_in_mode or 0) > 1
            )
        ]
        if not victims:
            audit_log.append(
                "cross_family_escape_floor_skipped:rank1_protected"
            )
            break
        victim = max(victims, key=lambda item: item.priority)
        victim_index = state.selected.index(victim)
        state.selected[victim_index] = context
        state.selection_modes.pop(victim.candidate.candidate_id, None)
        state.selection_modes[candidate_id] = context.mode or "explore"
        selected_cross_ids.add(candidate_id)
        replaced.append({
            "out": victim.candidate.candidate_id,
            "in": candidate_id,
        })

    if not added and not replaced:
        return None, tuple(audit_log)
    forced_escape = {
        "target_cross": target_cross,
        "added": added,
        "replaced": replaced,
    }
    audit_log.append(
        "forced_cross_family_escape_floor:"
        + ",".join(
            added
            + [
                f"{replacement['out']}->{replacement['in']}"
                for replacement in replaced
            ]
        )
    )
    return forced_escape, tuple(audit_log)


def rank_candidates(
    request: SelectionRankingRequest,
) -> SelectionRankingResult:
    """Select candidates in quota, backfill, and escape-floor order."""

    state = _SelectionState(remaining=list(request.eligible_candidates))
    all_contexts = (
        list(request.eligible_candidates)
        + list(request.cost_deferred_candidates)
        + list(request.route_deferred_candidates)
        + list(request.cross_family_escape_candidates)
    )
    context_by_id = {
        context.candidate.candidate_id: context
        for context in all_contexts
    }

    if request.global_priority_used:
        _select_global_priority(request, state)
    else:
        _select_mode_quotas(request, state)
        _backfill_from(request, state, state.remaining)
        _backfill_from(request, state, request.cost_deferred_candidates)
        _backfill_from(request, state, request.route_deferred_candidates)

    forced_escape, audit_log = _apply_cross_family_escape_floor(
        request,
        state,
        context_by_id,
    )
    return SelectionRankingResult(
        selected_candidates=tuple(
            context.candidate
            for context in state.selected
        ),
        selection_modes=dict(state.selection_modes),
        rejected_per_mode={
            mode: tuple(state.rejected_per_mode[mode])
            for mode in MODES
        },
        global_priority_backfill_ids=tuple(
            state.global_priority_backfill_ids
        ),
        forced_cross_family_escape_floor=forced_escape,
        audit_log=audit_log,
    )
