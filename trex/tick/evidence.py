"""Pure evidence reduction for one live campaign tick.

Clustering is performed by :mod:`trex.tick.evidence_clustering`; this module
turns those bins plus execution history into an ``EvidenceSummary`` and an
optional production-panel snapshot.  It performs no archive writes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from ..evidence.attribution import is_canonical_su_record
from ..evidence_reducer import reduce_evidence
from ..planner import diagnostic_driver_tldr
from ..schemas import (
    ActionCandidate,
    EvidenceSummary,
    HypothesisCard,
    LLMHealthSummary,
    PanelSelection,
    ResultRecord,
    RouteHealthSummary,
    TargetConstraint,
)
from ..success_criteria import is_near_miss
from .evidence_clustering import ResultClusteringResult


STRICT_SU_COLLAPSE_MIN_COUNT = 4


@dataclass(frozen=True)
class EvidenceAssemblyRequest:
    """All explicit scientific and runtime inputs to evidence reduction."""

    target: TargetConstraint
    tick_id: str
    tick_id_int: int
    elapsed_wall_h: float
    remaining_wall_h: float
    pending_children: int
    inflight_gpu_h: float
    results: list[ResultRecord]
    hypotheses: list[HypothesisCard]
    spawning_actions: dict[str, ActionCandidate]
    prior_evidence: list[EvidenceSummary]
    clustering: ResultClusteringResult
    window_size: int
    worker_wall_gpu_count: int
    planner_model: str
    enable_exemplars: bool
    route_health_summary: RouteHealthSummary
    recent_fallback_rate: float
    charged_gpu_count: float | None
    charged_gpu_h_total: float | None
    charged_gpu_h_recent: float | None
    charged_gpu_h_scope: str
    recent_ticks_history: list[dict[str, Any]]
    diagnostic_chain_backlog: dict[str, Any]
    llm_health: LLMHealthSummary
    pending_family_load: dict[str, Any]
    dispatch_realization: dict[str, Any]
    execution_realization: dict[str, Any]


@dataclass(frozen=True)
class EvidenceAssemblyResult:
    """Evidence records and stable counters used by user-facing summaries."""

    evidence: EvidenceSummary
    production_panel_record: PanelSelection | None
    all_result_count: int
    window_result_count: int
    strict_total: int
    strict_window: int


def _su_bin_key(result: ResultRecord) -> str | None:
    value = result.bins.get("foldseek_su") if result.bins else None
    return str(value) if value else None


def _tick_to_int(tick_id: str | None) -> int | None:
    if not tick_id:
        return None
    if tick_id.startswith("v7r") and tick_id[3:].isdigit():
        return int(tick_id[3:])
    if tick_id.startswith("t_") and tick_id[2:].isdigit():
        return int(tick_id[2:])
    if tick_id.isdigit():
        return int(tick_id)
    return None


def assemble_tick_evidence(
    request: EvidenceAssemblyRequest,
) -> EvidenceAssemblyResult:
    """Reduce clustered results into the immutable live-tick evidence record."""

    results = request.results
    clustering = request.clustering
    window_results = results[-request.window_size:]

    def is_strict(result: ResultRecord) -> bool:
        return is_canonical_su_record(result, request.spawning_actions)

    def su_bins(records: list[ResultRecord]) -> set[str]:
        return {
            key
            for result in records
            if is_strict(result)
            for key in [_su_bin_key(result)]
            if key is not None
        }

    def sequence_bins(records: list[ResultRecord]) -> set[str]:
        return {
            str(value)
            for result in records
            if is_strict(result)
            for value in [
                result.bins.get("sequence_su") if result.bins else None
            ]
            if value
        }

    def strict_records(records: list[ResultRecord]) -> list[ResultRecord]:
        return [result for result in records if is_strict(result)]

    strict_window = sum(1 for result in window_results if is_strict(result))
    strict_total = sum(1 for result in results if is_strict(result))

    structure_bin_counts: dict[str, int] = {}
    for result in window_results:
        structure_bin = result.bins.get("foldseek") if result.bins else None
        if structure_bin:
            structure_bin_counts[structure_bin] = (
                structure_bin_counts.get(structure_bin, 0) + 1
            )
    binned_result_count = sum(structure_bin_counts.values())
    duplicate_fraction = (
        1.0 - len(structure_bin_counts) / max(1, binned_result_count)
        if structure_bin_counts
        else None
    )
    top_bin_share = (
        max(structure_bin_counts.values()) / max(1, binned_result_count)
        if structure_bin_counts
        else None
    )

    near_miss_bins: set[str] = set()
    for result in window_results:
        if result.exit_status != "ok" or not is_near_miss(result.metrics):
            continue
        bins = result.bins or {}
        near_miss_bins.add(
            bins.get("foldseek_near_miss")
            or bins.get("foldseek")
            or bins.get("refilter_source")
            or result.result_id
        )
    near_miss_count = len(near_miss_bins)

    panel_ready_bins: dict[str, set[str]] = {}
    for result in results:
        if not result.panel_ready or not result.bins:
            continue
        for bin_name, bin_value in result.bins.items():
            if bin_name in {
                "foldseek_su",
                "sequence_su",
                "foldseek_near_miss",
            }:
                continue
            panel_ready_bins.setdefault(bin_name, set()).add(bin_value)
    panel_ready_bins_covered = sum(
        len(bin_values) for bin_values in panel_ready_bins.values()
    )

    run_su_count = len(su_bins(results))
    pre_window_results = (
        results[:-request.window_size]
        if len(results) > request.window_size
        else []
    )
    run_su_count_delta = max(
        0,
        run_su_count - len(su_bins(pre_window_results)),
    )

    strict_su_window_counts: dict[str, int] = {}
    for result in strict_records(window_results):
        structure_su_bin = (result.bins or {}).get("foldseek_su")
        if structure_su_bin:
            strict_su_window_counts[structure_su_bin] = (
                strict_su_window_counts.get(structure_su_bin, 0) + 1
            )
    strict_su_top_bin_share: float | None = None
    strict_su_window_total = sum(strict_su_window_counts.values())
    if strict_su_window_total >= STRICT_SU_COLLAPSE_MIN_COUNT:
        strict_su_top_bin_share = (
            max(strict_su_window_counts.values())
            / max(1, strict_su_window_total)
        )

    seq_unique_strict_count = clustering.seq_unique_strict_count
    seq_unique_strict_delta = clustering.seq_unique_strict_delta
    joint_struct_seq_unique_count = clustering.joint_struct_seq_unique_count
    seq_duplicate_fraction = clustering.seq_duplicate_fraction
    top_seq_bin_share = clustering.top_seq_bin_share
    if (
        clustering.strict_result_ids
        and clustering.sequence_dedup_status == "ok"
    ):
        all_sequence_bins = sequence_bins(results)
        sequence_binned_count = sum(
            1
            for result in strict_records(results)
            if result.bins and result.bins.get("sequence_su")
        )
        sequence_has_full_coverage = (
            sequence_binned_count == len(clustering.strict_result_ids)
        )
        if sequence_has_full_coverage:
            seq_unique_strict_count = len(all_sequence_bins)
            seq_unique_strict_delta = max(
                0,
                len(all_sequence_bins) - len(sequence_bins(pre_window_results)),
            )
            if (
                clustering.foldseek_su_status == "ok"
                and clustering.foldseek_su_coverage is not None
                and clustering.foldseek_su_coverage >= 0.999
            ):
                structure_sequence_pairs = {
                    (
                        (result.bins or {}).get("foldseek_su"),
                        (result.bins or {}).get("sequence_su"),
                    )
                    for result in strict_records(results)
                    if (result.bins or {}).get("foldseek_su")
                    and (result.bins or {}).get("sequence_su")
                }
                joint_struct_seq_unique_count = len(structure_sequence_pairs)

        sequence_window_counts: dict[str, int] = {}
        for result in strict_records(window_results):
            sequence_bin = (
                result.bins.get("sequence_su") if result.bins else None
            )
            if sequence_bin:
                sequence_window_counts[sequence_bin] = (
                    sequence_window_counts.get(sequence_bin, 0) + 1
                )
        sequence_binned_window_count = sum(sequence_window_counts.values())
        if sequence_window_counts:
            sequence_repeats = sum(
                count
                for count in sequence_window_counts.values()
                if count > 1
            )
            seq_duplicate_fraction = (
                sequence_repeats / max(1, sequence_binned_window_count)
            )
            top_seq_bin_share = (
                max(sequence_window_counts.values())
                / max(1, sequence_binned_window_count)
            )

    production_panel_status = "not_run"
    production_panel_value: float | None = None
    production_panel_selected_ids: list[str] = []
    production_panel_diversity_bins: dict[str, int] = {}
    production_panel_gap_reasons: list[str] = []
    production_near_miss_ids: list[str] = []
    production_panel_record: PanelSelection | None = None
    panel_dedup_trusted = (
        clustering.foldseek_su_status in {"ok", "cached_ok"}
        and clustering.foldseek_su_coverage is not None
        and clustering.foldseek_su_coverage >= 0.999
    )
    try:
        from ..panel import (
            ProductionPanelConfig,
            best_near_miss_backups,
            select_production_panel,
        )

        panel_config = ProductionPanelConfig(
            K=request.target.panel_size_K,
            require_trusted_structure_bin=True,
        )
        production_panel_record = select_production_panel(
            results if panel_dedup_trusted else [],
            cfg=panel_config,
            panel_id=f"panel_snapshot_{request.tick_id}",
        )
        production_panel_selected_ids = list(production_panel_record.selected_ids)
        production_panel_value = production_panel_record.panel_value
        production_panel_diversity_bins = dict(
            production_panel_record.diversity_bins
        )
        production_panel_gap_reasons = list(
            production_panel_record.hard_gate_failures
        )
        production_near_miss_ids = best_near_miss_backups(
            results,
            K=min(4, request.target.panel_size_K),
            require_structure_artifact=panel_config.require_structure_artifact,
        )
        if clustering.strict_result_ids and not panel_dedup_trusted:
            production_panel_status = "degraded_untrusted_structure_dedup"
            dedup_reason = (
                "untrusted_structure_dedup:"
                f"status={clustering.foldseek_su_status},"
                f"coverage={clustering.foldseek_su_coverage}"
            )
            production_panel_gap_reasons.insert(0, dedup_reason)
            production_panel_record = replace(
                production_panel_record,
                hard_gate_failures=[
                    dedup_reason,
                    *production_panel_record.hard_gate_failures,
                ],
            )
        else:
            production_panel_status = (
                "ok" if production_panel_selected_ids else "no_eligible_strict"
            )
    except Exception as exc:  # noqa: BLE001
        production_panel_status = f"failed:{type(exc).__name__}"
        production_panel_gap_reasons = [str(exc)[:200]]

    worker_gpu_h_total = sum(result.gpu_h for result in results)
    scientific_su_seen: set[str] = set()
    scientific_su_hwm_gpu_h = 0.0
    hwm_tick = 0
    cumulative_gpu_h = 0.0
    for result in results:
        cumulative_gpu_h += float(result.gpu_h or 0.0)
        if not is_strict(result):
            continue
        su_key = _su_bin_key(result)
        if su_key is None or su_key in scientific_su_seen:
            continue
        scientific_su_seen.add(su_key)
        scientific_su_hwm_gpu_h = cumulative_gpu_h
        hwm_tick = _tick_to_int(result.tick_id) or hwm_tick

    previous_run_su_hwm = max(
        (
            int(
                getattr(previous, "run_su_hwm", None)
                or getattr(previous, "run_su_count", 0)
                or 0
            )
            for previous in request.prior_evidence
        ),
        default=0,
    )
    run_su_hwm = max(previous_run_su_hwm, run_su_count)
    run_su_hwm_delta = max(0, run_su_hwm - previous_run_su_hwm)
    worker_wall_gpu_count = float(max(1, int(request.worker_wall_gpu_count or 3)))
    worker_wall_gpu_h_total = max(
        0.0,
        float(request.elapsed_wall_h or 0.0) * worker_wall_gpu_count,
    )
    hwm_advanced_this_tick = bool(request.prior_evidence) and run_su_hwm_delta > 0
    new_scientific_su_this_tick = hwm_advanced_this_tick or (
        bool(scientific_su_seen)
        and (
            hwm_tick == request.tick_id_int
            or (
                hwm_tick == 0
                and run_su_count > previous_run_su_hwm
            )
        )
    )
    if hwm_advanced_this_tick:
        scientific_su_hwm_gpu_h = worker_gpu_h_total
        hwm_tick = request.tick_id_int
    dry_gpu_h_total = (
        worker_gpu_h_total
        if new_scientific_su_this_tick
        else worker_gpu_h_total + max(0.0, request.inflight_gpu_h)
    )
    gpu_h_since_last_su = max(
        0.0,
        dry_gpu_h_total - scientific_su_hwm_gpu_h,
    )
    ticks_since_last_su = max(0, request.tick_id_int - hwm_tick)

    evidence = reduce_evidence(
        tick_id=request.tick_id,
        target_id=request.target.target_id,
        target_class=request.target.target_class,
        elapsed_wall_h=request.elapsed_wall_h,
        remaining_wall_h=request.remaining_wall_h,
        pending_children=request.pending_children,
        worker_gpu_h_total=worker_gpu_h_total,
        worker_wall_gpu_count=worker_wall_gpu_count,
        worker_wall_gpu_h_total=worker_wall_gpu_h_total,
        run_su_hwm=run_su_hwm,
        run_su_hwm_delta=run_su_hwm_delta,
        gpu_h_since_last_su=gpu_h_since_last_su,
        ticks_since_last_su=ticks_since_last_su,
        all_results=results,
        window_results=window_results,
        run_su_count=run_su_count,
        run_su_count_delta=run_su_count_delta,
        duplicate_fraction=duplicate_fraction,
        near_miss_count=near_miss_count,
        top_bin_share=top_bin_share,
        panel_ready_count=sum(1 for result in results if result.panel_ready),
        panel_ready_bins_covered=panel_ready_bins_covered,
        llm_model=request.planner_model,
        spawning_actions=request.spawning_actions,
        route_health_summary=request.route_health_summary,
        recent_fallback_rate=request.recent_fallback_rate,
        enable_exemplars=request.enable_exemplars,
        foldseek_su_status=clustering.foldseek_su_status,
        foldseek_su_coverage=clustering.foldseek_su_coverage,
        strict_su_top_bin_share=strict_su_top_bin_share,
        strict_su_tm08_status=clustering.strict_su_tm08_status,
        strict_su_tm08_coverage=clustering.strict_su_tm08_coverage,
        strict_su_tm08_recent_count=clustering.strict_su_tm08_recent_count,
        strict_su_live_recent_count=clustering.strict_su_live_recent_count,
        strict_su_tm08_delta_vs_live=clustering.strict_su_tm08_delta_vs_live,
        strict_su_tm08_live_split_ratio=(
            clustering.strict_su_tm08_live_split_ratio
        ),
        strict_su_tm05_recent_count=clustering.strict_su_tm05_recent_count,
        strict_su_tm08_delta_vs_tm05=clustering.strict_su_tm08_delta_vs_tm05,
        strict_su_tm08_split_ratio=clustering.strict_su_tm08_split_ratio,
        strict_su_tm08_result_scope=clustering.strict_su_tm08_result_scope,
        structure_dedup_scope=clustering.structure_dedup_scope,
        structure_dedup_fallback_count=clustering.structure_dedup_fallback_count,
        foldseek_archive_status=clustering.foldseek_archive_status,
        foldseek_archive_coverage=clustering.foldseek_archive_coverage,
        foldseek_archive_result_scope=clustering.foldseek_archive_result_scope,
        whole_archive_structure_dedup_scope=(
            clustering.whole_archive_structure_dedup_scope
        ),
        whole_archive_structure_dedup_fallback_count=(
            clustering.whole_archive_structure_dedup_fallback_count
        ),
        sequence_dedup_status=clustering.sequence_dedup_status,
        sequence_dedup_coverage=clustering.sequence_dedup_coverage,
        near_miss_dedup_status=clustering.near_miss_dedup_status,
        near_miss_dedup_coverage=clustering.near_miss_dedup_coverage,
        near_miss_cluster_by_result_id=(
            clustering.near_miss_cluster_by_result_id
        ),
        seq_unique_strict_count=seq_unique_strict_count,
        seq_unique_strict_delta=seq_unique_strict_delta,
        joint_struct_seq_unique_count=joint_struct_seq_unique_count,
        seq_duplicate_fraction=seq_duplicate_fraction,
        top_seq_bin_share=top_seq_bin_share,
        charged_gpu_count=request.charged_gpu_count,
        charged_gpu_h_total=request.charged_gpu_h_total,
        charged_gpu_h_recent=request.charged_gpu_h_recent,
        charged_gpu_h_scope=request.charged_gpu_h_scope,
        production_panel_status=production_panel_status,
        production_panel_value=production_panel_value,
        production_panel_selected_ids=production_panel_selected_ids,
        production_panel_diversity_bins=production_panel_diversity_bins,
        production_panel_gap_reasons=production_panel_gap_reasons,
        production_near_miss_ids=production_near_miss_ids,
        hypotheses=request.hypotheses,
    )
    evidence = replace(
        evidence,
        recent_ticks_history=(
            request.recent_ticks_history
            if request.recent_ticks_history
            else evidence.recent_ticks_history
        ),
        diagnostic_chain_backlog=request.diagnostic_chain_backlog,
        llm_health=request.llm_health,
        pending_family_load=request.pending_family_load,
        dispatch_realization=request.dispatch_realization,
        execution_realization=request.execution_realization,
        diagnostic_driver_tldr=diagnostic_driver_tldr(evidence),
    )
    return EvidenceAssemblyResult(
        evidence=evidence,
        production_panel_record=production_panel_record,
        all_result_count=len(results),
        window_result_count=len(window_results),
        strict_total=strict_total,
        strict_window=strict_window,
    )


__all__ = [
    "EvidenceAssemblyRequest",
    "EvidenceAssemblyResult",
    "assemble_tick_evidence",
]
