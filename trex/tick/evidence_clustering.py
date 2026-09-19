"""Structural and sequence clustering for live-tick evidence.

This phase is the only evidence phase that mutates ``ResultRecord.bins``.  The
mutation is deliberate: downstream reducer, panel, and selector logic must all
observe the same cached Foldseek/MMseqs assignments during a tick.  It never
writes records to the archive.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..evidence.attribution import is_canonical_su_record
from ..schemas import ActionCandidate, EvidenceSummary, ResultRecord, TargetConstraint
from ..success_criteria import is_near_miss
from .config import FoldseekConfig, SequenceDedupConfig


@dataclass(frozen=True)
class ResultClusteringRequest:
    """Inputs required to assign live evidence-clustering bins."""

    results: list[ResultRecord]
    target: TargetConstraint
    spawning_actions: dict[str, ActionCandidate]
    previous_evidence: EvidenceSummary | None
    tick_id_int: int
    window_size: int
    foldseek_config: FoldseekConfig
    sequence_config: SequenceDedupConfig


@dataclass(frozen=True)
class ResultClusteringResult:
    """Deduplication provenance and diagnostics consumed by evidence assembly."""

    strict_result_ids: frozenset[str]
    foldseek_su_status: str = "disabled"
    foldseek_su_coverage: float | None = None
    structure_dedup_scope: str = "disabled"
    structure_dedup_fallback_count: int = 0
    foldseek_archive_status: str = "disabled"
    foldseek_archive_coverage: float | None = None
    foldseek_archive_result_scope: str = "lifetime_or_unknown"
    whole_archive_structure_dedup_scope: str = "disabled"
    whole_archive_structure_dedup_fallback_count: int = 0
    near_miss_dedup_status: str = "disabled"
    near_miss_dedup_coverage: float | None = None
    near_miss_cluster_by_result_id: dict[str, str] = field(default_factory=dict)
    strict_su_tm08_status: str = "disabled"
    strict_su_tm08_coverage: float | None = None
    strict_su_tm08_recent_count: int | None = None
    strict_su_live_recent_count: int | None = None
    strict_su_tm08_delta_vs_live: int | None = None
    strict_su_tm08_live_split_ratio: float | None = None
    strict_su_tm05_recent_count: int | None = None
    strict_su_tm08_delta_vs_tm05: int | None = None
    strict_su_tm08_split_ratio: float | None = None
    strict_su_tm08_result_scope: str = "disabled"
    sequence_dedup_status: str = "disabled"
    sequence_dedup_coverage: float | None = None
    seq_unique_strict_count: int | None = None
    seq_unique_strict_delta: int | None = None
    joint_struct_seq_unique_count: int | None = None
    seq_duplicate_fraction: float | None = None
    top_seq_bin_share: float | None = None


def _fallback_binder_chain_id(
    target: TargetConstraint,
    default: str = "B",
) -> str:
    target_chains = {
        str(chain).strip()
        for chain in (target.chain_ids or [])
        if str(chain).strip()
    }
    if default and default not in target_chains:
        return default
    for code in range(ord("A"), ord("Z") + 1):
        chain = chr(code)
        if chain not in target_chains:
            return chain
    return default or "B"


def cluster_evidence_results(
    request: ResultClusteringRequest,
) -> ResultClusteringResult:
    """Assign result bins and return complete deduplication provenance."""

    results = request.results
    target = request.target
    foldseek = request.foldseek_config
    sequence = request.sequence_config
    strict_result_ids = {
        result.result_id
        for result in results
        if is_canonical_su_record(result, request.spawning_actions)
    }
    near_miss_result_ids = {
        result.result_id
        for result in results
        if result.exit_status == "ok" and is_near_miss(result.metrics)
    }

    foldseek_su_status = "disabled"
    foldseek_su_coverage: float | None = None
    structure_dedup_scope = "disabled"
    structure_dedup_fallback_count = 0
    foldseek_archive_status = "disabled"
    foldseek_archive_coverage: float | None = None
    foldseek_archive_result_scope = "lifetime_or_unknown"
    whole_archive_structure_dedup_scope = "disabled"
    whole_archive_structure_dedup_fallback_count = 0
    near_miss_dedup_status = "disabled"
    near_miss_dedup_coverage: float | None = None
    near_miss_cluster_by_result_id: dict[str, str] = {}
    strict_su_tm08_status = "disabled"
    strict_su_tm08_coverage: float | None = None
    strict_su_tm08_recent_count: int | None = None
    strict_su_live_recent_count: int | None = None
    strict_su_tm08_delta_vs_live: int | None = None
    strict_su_tm08_live_split_ratio: float | None = None
    strict_su_tm05_recent_count: int | None = None
    strict_su_tm08_delta_vs_tm05: int | None = None
    strict_su_tm08_split_ratio: float | None = None
    strict_su_tm08_result_scope = "disabled"

    foldseek_binder_chain_id = _fallback_binder_chain_id(target)
    if foldseek.enabled and results:
        from ..clustering_cache import cluster_archive_pdbs_cached
        from ..foldseek_clusterer import _scored_pdb_path, apply_clusters_to_bins

        archive_window = max(foldseek.whole_archive_window, request.window_size)
        recent_scored_result_ids = [
            result.result_id
            for result in results
            if result.target_id == target.target_id
            and _scored_pdb_path(result) is not None
        ][-archive_window:]
        archive_clusters = cluster_archive_pdbs_cached(
            results,
            target_id=target.target_id,
            foldseek_binary=foldseek.binary,
            min_tm_score=foldseek.collapse_tm_score,
            timeout_seconds=foldseek.timeout_seconds,
            only_result_ids=set(recent_scored_result_ids),
            binder_chain_id=foldseek_binder_chain_id,
        )
        updated_count = apply_clusters_to_bins(
            results,
            archive_clusters.cluster_by_result_id,
        )
        foldseek_archive_status = archive_clusters.status
        foldseek_archive_coverage = (
            updated_count / archive_clusters.n_structures
            if archive_clusters.n_structures > 0
            else None
        )
        foldseek_archive_result_scope = f"recent_scored_window_{archive_window}"
        whole_archive_structure_dedup_scope = archive_clusters.structure_scope
        whole_archive_structure_dedup_fallback_count = archive_clusters.n_scope_fallback
        print(
            f"  [foldseek] status={archive_clusters.status} "
            f"scope={foldseek_archive_result_scope} "
            f"n_structures={archive_clusters.n_structures} "
            f"n_clusters={archive_clusters.n_clusters} "
            f"n_updated={updated_count} "
            f"struct_scope={archive_clusters.structure_scope} "
            f"scope_fallback={archive_clusters.n_scope_fallback}",
            flush=True,
        )
        if (
            archive_clusters.status not in {"ok", "no_structures"}
            and archive_clusters.stderr_tail
        ):
            print(
                f"  [foldseek] stderr_tail: {archive_clusters.stderr_tail[:200]}",
                flush=True,
            )

        if strict_result_ids:
            strict_clusters = cluster_archive_pdbs_cached(
                results,
                target_id=target.target_id,
                foldseek_binary=foldseek.binary,
                min_tm_score=foldseek.min_tm_score,
                timeout_seconds=foldseek.timeout_seconds,
                only_result_ids=strict_result_ids,
                binder_chain_id=foldseek_binder_chain_id,
            )
            strict_updated_count = apply_clusters_to_bins(
                results,
                strict_clusters.cluster_by_result_id,
                bin_key="foldseek_su",
            )
            foldseek_su_status = strict_clusters.status
            foldseek_su_coverage = len(
                strict_result_ids & set(strict_clusters.cluster_by_result_id)
            ) / len(strict_result_ids)
            structure_dedup_scope = strict_clusters.structure_scope
            structure_dedup_fallback_count = strict_clusters.n_scope_fallback
            print(
                f"  [foldseek:SU] strict_structures={strict_clusters.n_structures} "
                f"strict_clusters={strict_clusters.n_clusters} "
                f"status={strict_clusters.status} "
                f"n_updated={strict_updated_count} "
                f"coverage={foldseek_su_coverage:.2f} "
                f"scope={strict_clusters.structure_scope} "
                f"scope_fallback={strict_clusters.n_scope_fallback}",
                flush=True,
            )

            recent_strict_result_ids = [
                result.result_id
                for result in results
                if result.result_id in strict_result_ids
                and result.target_id == target.target_id
            ][-max(1, int(foldseek.fine_strict_window)):]
            recent_strict_set = set(recent_strict_result_ids)
            fine_every = max(1, int(foldseek.fine_refresh_every_ticks))
            previous_tm08_status = (
                getattr(
                    request.previous_evidence,
                    "strict_su_tm08_status",
                    "disabled",
                )
                if request.previous_evidence
                else "disabled"
            )
            fine_due = (
                request.previous_evidence is None
                or request.tick_id_int <= 2
                or request.tick_id_int % fine_every == 0
                or previous_tm08_status
                in {"disabled", "too_few_strict", "no_strict"}
            )
            if not foldseek.strict_tm08_diagnostic_enabled:
                strict_su_tm08_status = "disabled"
                strict_su_tm08_result_scope = "disabled"
            elif len(recent_strict_set) >= foldseek.fine_min_strict and fine_due:
                fine_clusters = cluster_archive_pdbs_cached(
                    results,
                    target_id=target.target_id,
                    foldseek_binary=foldseek.binary,
                    min_tm_score=foldseek.fine_tm_score,
                    timeout_seconds=foldseek.timeout_seconds,
                    only_result_ids=recent_strict_set,
                    binder_chain_id=foldseek_binder_chain_id,
                )
                fine_updated_count = apply_clusters_to_bins(
                    results,
                    fine_clusters.cluster_by_result_id,
                    bin_key="foldseek_su_tm08_recent",
                )
                strict_su_tm08_status = fine_clusters.status
                strict_su_tm08_coverage = (
                    len(recent_strict_set & set(fine_clusters.cluster_by_result_id))
                    / len(recent_strict_set)
                    if recent_strict_set
                    else None
                )
                live_recent_bins = {
                    (result.bins or {}).get("foldseek_su")
                    for result in results
                    if result.result_id in recent_strict_set
                    and (result.bins or {}).get("foldseek_su")
                }
                tm08_recent_bins = {
                    cluster_id
                    for result_id, cluster_id in fine_clusters.cluster_by_result_id.items()
                    if result_id in recent_strict_set
                }
                strict_su_tm05_recent_count = len(live_recent_bins)
                strict_su_live_recent_count = strict_su_tm05_recent_count
                strict_su_tm08_recent_count = len(tm08_recent_bins)
                if strict_su_tm05_recent_count:
                    strict_su_tm08_delta_vs_tm05 = (
                        strict_su_tm08_recent_count - strict_su_tm05_recent_count
                    )
                    strict_su_tm08_split_ratio = (
                        strict_su_tm08_recent_count / strict_su_tm05_recent_count
                    )
                    strict_su_tm08_delta_vs_live = strict_su_tm08_delta_vs_tm05
                    strict_su_tm08_live_split_ratio = strict_su_tm08_split_ratio
                strict_su_tm08_result_scope = (
                    f"recent_strict_window_{len(recent_strict_set)}"
                )
                print(
                    f"  [foldseek:SU_TM08_recent] "
                    f"strict_structures={fine_clusters.n_structures} "
                    f"tm08_clusters={fine_clusters.n_clusters} "
                    f"status={fine_clusters.status} "
                    f"n_updated={fine_updated_count} "
                    f"coverage={strict_su_tm08_coverage:.2f} "
                    f"live_clusters={strict_su_tm05_recent_count} "
                    f"split_ratio={strict_su_tm08_split_ratio}",
                    flush=True,
                )
            elif (
                len(recent_strict_set) >= foldseek.fine_min_strict
                and request.previous_evidence is not None
            ):
                previous = request.previous_evidence
                strict_su_tm08_status = (
                    "cached_ok"
                    if previous_tm08_status in {"ok", "cached_ok"}
                    else f"cached_{previous_tm08_status}"
                )
                strict_su_tm08_coverage = getattr(
                    previous, "strict_su_tm08_coverage", None
                )
                strict_su_tm05_recent_count = getattr(
                    previous, "strict_su_tm05_recent_count", None
                )
                strict_su_live_recent_count = getattr(
                    previous,
                    "strict_su_live_recent_count",
                    strict_su_tm05_recent_count,
                )
                strict_su_tm08_recent_count = getattr(
                    previous, "strict_su_tm08_recent_count", None
                )
                strict_su_tm08_delta_vs_tm05 = getattr(
                    previous, "strict_su_tm08_delta_vs_tm05", None
                )
                strict_su_tm08_delta_vs_live = getattr(
                    previous,
                    "strict_su_tm08_delta_vs_live",
                    strict_su_tm08_delta_vs_tm05,
                )
                strict_su_tm08_split_ratio = getattr(
                    previous, "strict_su_tm08_split_ratio", None
                )
                strict_su_tm08_live_split_ratio = getattr(
                    previous,
                    "strict_su_tm08_live_split_ratio",
                    strict_su_tm08_split_ratio,
                )
                strict_su_tm08_result_scope = "cached:" + str(
                    getattr(previous, "strict_su_tm08_result_scope", "unknown")
                )
                print(
                    f"  [foldseek:SU_TM08_recent] "
                    f"cached status={strict_su_tm08_status} "
                    f"tm08_clusters={strict_su_tm08_recent_count} "
                    f"live_clusters={strict_su_tm05_recent_count} "
                    f"split_ratio={strict_su_tm08_split_ratio}",
                    flush=True,
                )
            else:
                strict_su_tm08_status = "too_few_strict"
                strict_su_tm08_coverage = 1.0
                strict_su_tm08_result_scope = (
                    f"recent_strict_window_{len(recent_strict_set)}"
                )
        else:
            foldseek_su_status = "no_strict"
            structure_dedup_scope = whole_archive_structure_dedup_scope
            structure_dedup_fallback_count = (
                whole_archive_structure_dedup_fallback_count
            )

        if near_miss_result_ids:
            near_miss_clusters = cluster_archive_pdbs_cached(
                results,
                target_id=target.target_id,
                foldseek_binary=foldseek.binary,
                min_tm_score=foldseek.collapse_tm_score,
                timeout_seconds=foldseek.timeout_seconds,
                only_result_ids=near_miss_result_ids,
                binder_chain_id=foldseek_binder_chain_id,
            )
            near_miss_cluster_by_result_id = dict(
                near_miss_clusters.cluster_by_result_id
            )
            near_miss_updated_count = apply_clusters_to_bins(
                results,
                near_miss_cluster_by_result_id,
                bin_key="foldseek_near_miss",
            )
            near_miss_dedup_status = near_miss_clusters.status
            near_miss_dedup_coverage = len(
                near_miss_result_ids & set(near_miss_cluster_by_result_id)
            ) / len(near_miss_result_ids)
            print(
                f"  [foldseek:near_miss] "
                f"near_structures={near_miss_clusters.n_structures} "
                f"near_clusters={near_miss_clusters.n_clusters} "
                f"status={near_miss_clusters.status} "
                f"n_updated={near_miss_updated_count} "
                f"coverage={near_miss_dedup_coverage:.2f} "
                f"scope={near_miss_clusters.structure_scope} "
                f"scope_fallback={near_miss_clusters.n_scope_fallback}",
                flush=True,
            )
        else:
            near_miss_dedup_status = "no_near_miss"
            near_miss_dedup_coverage = 1.0

    sequence_dedup_status = "disabled"
    sequence_dedup_coverage: float | None = None
    seq_unique_strict_count: int | None = None
    seq_unique_strict_delta: int | None = None
    joint_struct_seq_unique_count: int | None = None
    seq_duplicate_fraction: float | None = None
    top_seq_bin_share: float | None = None
    if sequence.enabled:
        if strict_result_ids:
            from ..clustering_cache import cluster_archive_sequences_cached
            from ..sequence_clusterer import apply_sequence_clusters_to_bins

            sequence_clusters = cluster_archive_sequences_cached(
                results,
                target_id=target.target_id,
                mmseqs_binary=sequence.binary,
                min_seq_id=sequence.min_seq_id,
                coverage=sequence.coverage,
                timeout_seconds=sequence.timeout_seconds,
                only_result_ids=strict_result_ids,
                binder_chain_id=_fallback_binder_chain_id(
                    target,
                    sequence.binder_chain_id,
                ),
            )
            sequence_updated_count = apply_sequence_clusters_to_bins(
                results,
                sequence_clusters.cluster_by_result_id,
                bin_key="sequence_su",
            )
            sequence_dedup_status = sequence_clusters.status
            sequence_dedup_coverage = len(
                strict_result_ids & set(sequence_clusters.cluster_by_result_id)
            ) / len(strict_result_ids)
            print(
                f"  [mmseqs:sequence_su] "
                f"strict_sequences={sequence_clusters.n_sequences} "
                f"seq_clusters={sequence_clusters.n_clusters} "
                f"status={sequence_clusters.status} "
                f"n_updated={sequence_updated_count} "
                f"coverage={sequence_dedup_coverage:.2f}",
                flush=True,
            )
            if (
                sequence_clusters.status not in {"ok", "no_sequences"}
                and sequence_clusters.stderr_tail
            ):
                print(
                    f"  [mmseqs] stderr_tail: "
                    f"{sequence_clusters.stderr_tail[:200]}",
                    flush=True,
                )
        else:
            sequence_dedup_status = "no_strict"
            sequence_dedup_coverage = 1.0
            seq_unique_strict_count = 0
            seq_unique_strict_delta = 0
            joint_struct_seq_unique_count = 0

    return ResultClusteringResult(
        strict_result_ids=frozenset(strict_result_ids),
        foldseek_su_status=foldseek_su_status,
        foldseek_su_coverage=foldseek_su_coverage,
        structure_dedup_scope=structure_dedup_scope,
        structure_dedup_fallback_count=structure_dedup_fallback_count,
        foldseek_archive_status=foldseek_archive_status,
        foldseek_archive_coverage=foldseek_archive_coverage,
        foldseek_archive_result_scope=foldseek_archive_result_scope,
        whole_archive_structure_dedup_scope=whole_archive_structure_dedup_scope,
        whole_archive_structure_dedup_fallback_count=(
            whole_archive_structure_dedup_fallback_count
        ),
        near_miss_dedup_status=near_miss_dedup_status,
        near_miss_dedup_coverage=near_miss_dedup_coverage,
        near_miss_cluster_by_result_id=near_miss_cluster_by_result_id,
        strict_su_tm08_status=strict_su_tm08_status,
        strict_su_tm08_coverage=strict_su_tm08_coverage,
        strict_su_tm08_recent_count=strict_su_tm08_recent_count,
        strict_su_live_recent_count=strict_su_live_recent_count,
        strict_su_tm08_delta_vs_live=strict_su_tm08_delta_vs_live,
        strict_su_tm08_live_split_ratio=strict_su_tm08_live_split_ratio,
        strict_su_tm05_recent_count=strict_su_tm05_recent_count,
        strict_su_tm08_delta_vs_tm05=strict_su_tm08_delta_vs_tm05,
        strict_su_tm08_split_ratio=strict_su_tm08_split_ratio,
        strict_su_tm08_result_scope=strict_su_tm08_result_scope,
        sequence_dedup_status=sequence_dedup_status,
        sequence_dedup_coverage=sequence_dedup_coverage,
        seq_unique_strict_count=seq_unique_strict_count,
        seq_unique_strict_delta=seq_unique_strict_delta,
        joint_struct_seq_unique_count=joint_struct_seq_unique_count,
        seq_duplicate_fraction=seq_duplicate_fraction,
        top_seq_bin_share=top_seq_bin_share,
    )


__all__ = [
    "ResultClusteringRequest",
    "ResultClusteringResult",
    "cluster_evidence_results",
]
