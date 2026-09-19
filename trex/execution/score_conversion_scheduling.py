"""Build canonical score-conversion follow-up candidates.

Scientific signal functions live in :mod:`trex.score_conversion`.  This module
owns the orchestration boundary that turns eligible diagnostic results into
explicit, inspectable ``ActionCandidate`` records.  It does not write to the
archive or launch workers.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Protocol, Sequence

from ..backend_extensions import get_backend_adapter
from ..capability_registry import Capability, CapabilityRegistry, default_registry
from ..refilter_roles import CANONICAL_SCORE_CONVERSION
from ..schemas import ActionCandidate, ResultRecord
from ..score_conversion import (
    is_score_convertible_diagnostic_record,
    native_strict_like_pending_score_conversion,
)


DIRECT_COMPLEXA_METHOD_FAMILIES = frozenset(
    {
        "complexa_beam",
        "complexa_best_of_n",
        "complexa_fk_steering",
        "complexa_mcts",
    }
)

ParentResultScore = Callable[[ResultRecord], float]
ScoredParentResult = tuple[float, ResultRecord]


class CapabilityLookup(Protocol):
    """Small registry surface required by the scheduling boundary."""

    def get(self, family: str) -> Capability | None: ...


@dataclass(frozen=True)
class ScoreConversionSchedulingRequest:
    """Complete input required to build score-conversion follow-up work."""

    source_candidate: ActionCandidate
    result_records: tuple[ResultRecord, ...]
    tick_id: str
    starting_sequence_number: int


@dataclass(frozen=True)
class ScoreConversionSchedule:
    """Inspectable output of one deterministic scheduling pass."""

    scheduled_candidates: tuple[ActionCandidate, ...]
    final_sequence_number: int
    eligible_parent_result_count: int
    selected_parent_result_count: int
    unavailable_downstream_method_families: tuple[str, ...] = ()
    skip_reason: str | None = None

    @property
    def scheduled_candidate_count(self) -> int:
        return len(self.scheduled_candidates)


@lru_cache(maxsize=1)
def _default_capability_registry() -> CapabilityRegistry:
    """Construct the default lookup once for repeated eligibility checks."""

    return default_registry()


def canonical_strict_metrics_present(result_record: ResultRecord) -> bool:
    """Return whether all official AF2 strict-score axes are numeric."""

    metrics = result_record.metrics or {}
    return all(
        isinstance(metrics.get(metric_name), (int, float))
        for metric_name in ("pLDDT", "iPAE", "binder_scRMSD")
    )


def is_canonical_score_conversion_result(result_record: ResultRecord) -> bool:
    """Return whether a result already came from canonical score conversion."""

    bins = result_record.bins or {}
    return (
        result_record.backend_family == "structure_refilter"
        and bins.get("refilter_role") == CANONICAL_SCORE_CONVERSION
        and bool(bins.get("refilter_source"))
    )


def result_requires_score_conversion(
    source_method_family: str,
    result_record: ResultRecord,
    *,
    capability_registry: CapabilityLookup | None = None,
) -> bool:
    """Return whether a generated structure still needs official AF2 scoring."""

    if result_record.exit_status != "ok":
        return False
    if is_canonical_score_conversion_result(result_record):
        return False
    if source_method_family == "structure_refilter":
        return False

    artifacts = result_record.artifacts or {}
    if not (
        artifacts.get("pdb_path")
        or artifacts.get("pdb_dir")
        or artifacts.get("cif_path")
    ):
        return False
    if not is_score_convertible_diagnostic_record(result_record):
        return False
    if source_method_family in DIRECT_COMPLEXA_METHOD_FAMILIES:
        return not canonical_strict_metrics_present(result_record)

    registry = capability_registry or _default_capability_registry()
    source_capability = registry.get(source_method_family)
    if source_capability is not None and source_capability.outputs_diagnostic_only:
        return True

    backend_adapter = get_backend_adapter(source_method_family)
    return bool(
        backend_adapter is not None
        and backend_adapter.capability.outputs_diagnostic_only
    )


def results_require_score_conversion(
    source_method_family: str,
    result_records: Sequence[ResultRecord],
    *,
    capability_registry: CapabilityLookup | None = None,
) -> bool:
    """Return whether at least one result requires canonical score conversion."""

    return any(
        result_requires_score_conversion(
            source_method_family,
            result_record,
            capability_registry=capability_registry,
        )
        for result_record in result_records
    )


def score_conversion_parent_identity(
    result_record: ResultRecord,
) -> tuple[str, str]:
    """Build a conservative identity used to avoid redundant AF2 scoring."""

    artifacts = result_record.artifacts or {}
    bins = result_record.bins or {}
    structure_path = artifacts.get("pdb_path") or artifacts.get("cif_path")
    if structure_path:
        return (
            result_record.backend_family,
            f"path:{str(Path(structure_path))}",
        )
    structure_directory = artifacts.get("pdb_dir")
    if structure_directory:
        return (
            result_record.backend_family,
            f"dir:{str(Path(structure_directory))}:{result_record.result_id}",
        )
    for sequence_key in (
        "sequence",
        "binder_sequence",
        "mpnn_sequence",
        "designed_sequence",
    ):
        sequence = bins.get(sequence_key) or artifacts.get(sequence_key)
        if sequence:
            return (
                result_record.backend_family,
                f"seq:{str(sequence).strip().upper()}",
            )
    return (
        result_record.backend_family,
        f"result:{result_record.result_id}",
    )


def deduplicate_scored_parent_results(
    scored_parent_results: Sequence[ScoredParentResult],
) -> list[ScoredParentResult]:
    """Keep the highest-scoring result for each score-conversion identity."""

    best_by_identity: dict[tuple[str, str], ScoredParentResult] = {}
    for score, result_record in scored_parent_results:
        identity = score_conversion_parent_identity(result_record)
        previous = best_by_identity.get(identity)
        if (
            previous is None
            or float(score) > float(previous[0])
            or (
                score == previous[0]
                and result_record.result_id < previous[1].result_id
            )
        ):
            best_by_identity[identity] = (score, result_record)
    return list(best_by_identity.values())


def score_conversion_candidate_limit(
    source_method_family: str,
    eligible_parent_results: Sequence[ResultRecord],
    *,
    capability_registry: CapabilityLookup | None = None,
) -> int:
    """Return how many eligible diagnostic results should receive AF2 scoring."""

    if not eligible_parent_results:
        return 0
    if (
        source_method_family in DIRECT_COMPLEXA_METHOD_FAMILIES
        and not results_require_score_conversion(
            source_method_family,
            eligible_parent_results,
            capability_registry=capability_registry,
        )
    ):
        return 0
    return len(eligible_parent_results)


def select_score_conversion_parents(
    scored_parent_results: Sequence[ScoredParentResult],
    candidate_limit: int,
) -> list[ScoredParentResult]:
    """Select proxy-ranked parents while retaining coverage of the ranked tail."""

    if candidate_limit <= 0 or not scored_parent_results:
        return []
    deduplicated_results = deduplicate_scored_parent_results(
        scored_parent_results
    )
    ranked_results = sorted(
        deduplicated_results,
        key=lambda pair: (
            not native_strict_like_pending_score_conversion(pair[1]),
            -float(pair[0]),
            pair[1].result_id,
        ),
    )
    if candidate_limit >= len(ranked_results):
        return list(ranked_results)

    top_result_count = max(
        1,
        min(candidate_limit, (candidate_limit + 1) // 2),
    )
    selected_results = list(ranked_results[:top_result_count])
    selected_result_ids = {
        result_record.result_id for _, result_record in selected_results
    }
    remaining_slot_count = candidate_limit - len(selected_results)
    remaining_results = ranked_results[top_result_count:]
    if remaining_slot_count > 0 and remaining_results:
        denominator = max(1, remaining_slot_count - 1)
        last_result_index = max(0, len(remaining_results) - 1)
        for slot_index in range(remaining_slot_count):
            result_index = (
                round((slot_index * last_result_index) / denominator)
                if remaining_slot_count > 1
                else last_result_index // 2
            )
            scored_result = remaining_results[result_index]
            result_id = scored_result[1].result_id
            if result_id not in selected_result_ids:
                selected_results.append(scored_result)
                selected_result_ids.add(result_id)

    if len(selected_results) < candidate_limit:
        for scored_result in ranked_results:
            result_id = scored_result[1].result_id
            if result_id in selected_result_ids:
                continue
            selected_results.append(scored_result)
            selected_result_ids.add(result_id)
            if len(selected_results) >= candidate_limit:
                break
    return selected_results[:candidate_limit]


def _empty_schedule(
    request: ScoreConversionSchedulingRequest,
    *,
    skip_reason: str,
    eligible_parent_result_count: int = 0,
    selected_parent_result_count: int = 0,
    unavailable_downstream_method_families: tuple[str, ...] = (),
) -> ScoreConversionSchedule:
    return ScoreConversionSchedule(
        scheduled_candidates=(),
        final_sequence_number=request.starting_sequence_number,
        eligible_parent_result_count=eligible_parent_result_count,
        selected_parent_result_count=selected_parent_result_count,
        unavailable_downstream_method_families=(
            unavailable_downstream_method_families
        ),
        skip_reason=skip_reason,
    )


def build_score_conversion_schedule(
    request: ScoreConversionSchedulingRequest,
    *,
    score_parent_result: ParentResultScore,
    capability_registry: CapabilityLookup | None = None,
) -> ScoreConversionSchedule:
    """Create deterministic follow-up candidates without archive side effects."""

    source_candidate = request.source_candidate
    if not request.result_records:
        return _empty_schedule(request, skip_reason="no_result_records")
    if not source_candidate.downstream_route_plan:
        return _empty_schedule(request, skip_reason="no_downstream_routes")

    registry = capability_registry or _default_capability_registry()
    source_capability = registry.get(source_candidate.method_family)
    if source_capability is None:
        return _empty_schedule(request, skip_reason="unknown_source_method_family")

    requires_conversion = results_require_score_conversion(
        source_candidate.method_family,
        request.result_records,
        capability_registry=registry,
    )
    if not (source_capability.outputs_diagnostic_only or requires_conversion):
        return _empty_schedule(
            request,
            skip_reason="canonical_score_conversion_not_required",
        )

    scored_parent_results = [
        (score_parent_result(result_record), result_record)
        for result_record in request.result_records
        if result_requires_score_conversion(
            source_candidate.method_family,
            result_record,
            capability_registry=registry,
        )
    ]
    eligible_parent_results = [
        result_record for _, result_record in scored_parent_results
    ]
    if not eligible_parent_results:
        return _empty_schedule(
            request,
            skip_reason="no_eligible_parent_results",
        )

    candidate_limit = score_conversion_candidate_limit(
        source_candidate.method_family,
        eligible_parent_results,
        capability_registry=registry,
    )
    selected_parent_results = select_score_conversion_parents(
        scored_parent_results,
        candidate_limit,
    )
    if not selected_parent_results:
        return _empty_schedule(
            request,
            skip_reason="no_selected_parent_results",
            eligible_parent_result_count=len(eligible_parent_results),
        )

    sequence_number = request.starting_sequence_number
    scheduled_candidates: list[ActionCandidate] = []
    unavailable_downstream_method_families: list[str] = []
    for downstream_method_family in source_candidate.downstream_route_plan:
        downstream_capability = registry.get(downstream_method_family)
        if (
            downstream_capability is None
            or downstream_capability.availability != "available"
        ):
            unavailable_downstream_method_families.append(
                downstream_method_family
            )
            continue

        for _, parent_result in selected_parent_results:
            sequence_number += 1
            scheduled_candidates.append(
                ActionCandidate(
                    candidate_id=(
                        f"chain_{request.tick_id}_{source_candidate.method_family}"
                        f"_to_{downstream_method_family}_{sequence_number:03d}"
                    ),
                    hypothesis_ids=list(source_candidate.hypothesis_ids),
                    parent_result_id=parent_result.result_id,
                    method_family=downstream_method_family,
                    operator_id=downstream_capability.default_operator_id,
                    lane_id=downstream_capability.default_lane_id,
                    config_delta={},
                    downstream_route_plan=[],
                    estimated_cost_class=(
                        downstream_capability.default_cost_class
                    ),
                    expected_signal=(
                        f"auto_chain:{source_candidate.method_family}->"
                        f"{downstream_method_family} "
                        f"parent={parent_result.result_id} "
                        f"role={CANONICAL_SCORE_CONVERSION}"
                    ),
                    evidence_refs=[parent_result.result_id],
                    feasibility=source_candidate.feasibility,
                    baseline_result_id=(
                        source_candidate.baseline_result_id
                        or source_candidate.parent_result_id
                    ),
                    refilter_role=(
                        CANONICAL_SCORE_CONVERSION
                        if downstream_method_family == "structure_refilter"
                        else None
                    ),
                )
            )

    unavailable_families = tuple(unavailable_downstream_method_families)
    if not scheduled_candidates:
        return _empty_schedule(
            request,
            skip_reason="no_available_downstream_method_family",
            eligible_parent_result_count=len(eligible_parent_results),
            selected_parent_result_count=len(selected_parent_results),
            unavailable_downstream_method_families=unavailable_families,
        )
    return ScoreConversionSchedule(
        scheduled_candidates=tuple(scheduled_candidates),
        final_sequence_number=sequence_number,
        eligible_parent_result_count=len(eligible_parent_results),
        selected_parent_result_count=len(selected_parent_results),
        unavailable_downstream_method_families=unavailable_families,
    )
