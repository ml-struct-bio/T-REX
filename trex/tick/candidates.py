"""Candidate construction and deterministic feasibility filtering phase."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from ..candidate_builder import BuilderConfig, build_candidates
from ..schemas import (
    ActionCandidate,
    EvidenceSummary,
    HypothesisCard,
    LaunchDecision,
    ResultRecord,
)


@dataclass(frozen=True)
class CandidatePhaseRequest:
    """Scientific and archive state needed to construct this tick's candidates."""

    hypotheses: Sequence[HypothesisCard]
    evidence: EvidenceSummary
    builder_config: BuilderConfig
    include_warmstart: bool
    warmstart_completed_families: frozenset[str]
    results: Sequence[ResultRecord]
    existing_actions: Sequence[ActionCandidate]
    prior_launches: Sequence[LaunchDecision]
    spawning_actions: Mapping[str, ActionCandidate]


@dataclass(frozen=True)
class CandidatePhaseResult:
    """Inspectable candidate populations produced before selection."""

    candidates: tuple[ActionCandidate, ...]
    feasible_candidates: tuple[ActionCandidate, ...]
    pending_chain_candidates: tuple[ActionCandidate, ...]
    parent_structure_available: bool
    structure_refilter_result_ids: frozenset[str]
    structure_refilter_scored_source_ids: frozenset[str]

    @property
    def infeasible_candidate_count(self) -> int:
        return len(self.candidates) - len(self.feasible_candidates)


def _result_has_usable_structure(result: ResultRecord) -> bool:
    artifacts = result.artifacts or {}
    for artifact_key in ("pdb_path", "cif_path"):
        artifact_path = artifacts.get(artifact_key)
        if artifact_path and Path(artifact_path).exists():
            return True

    structure_directory = artifacts.get("pdb_dir")
    if not structure_directory or not Path(structure_directory).exists():
        return False
    directory_path = Path(structure_directory)
    return (
        any(directory_path.glob("*.pdb"))
        or any(directory_path.glob("*.cif"))
        or any(directory_path.glob("*.mmcif"))
    )


def build_candidate_phase(request: CandidatePhaseRequest) -> CandidatePhaseResult:
    """Build candidates and expose each deterministic filtering population."""

    parent_structure_available = any(
        _result_has_usable_structure(result) for result in request.results
    )
    structure_refilter_result_ids = {
        result.result_id
        for result in request.results
        if result.backend_family == "structure_refilter"
    }
    known_result_ids = {result.result_id for result in request.results}
    structure_refilter_scored_source_ids: set[str] = set()
    for result in request.results:
        if result.backend_family != "structure_refilter":
            continue
        bins = result.bins or {}
        possible_sources = [
            bins.get("refilter_source"),
            result.parent_ids[1] if len(result.parent_ids or []) >= 2 else None,
        ]
        spawning_action = request.spawning_actions.get(result.result_id)
        if spawning_action is not None:
            possible_sources.append(spawning_action.parent_result_id)
        for source_result_id in possible_sources:
            if source_result_id and source_result_id in known_result_ids:
                structure_refilter_scored_source_ids.add(source_result_id)

    candidates = tuple(build_candidates(
        list(request.hypotheses),
        request.evidence,
        cfg=request.builder_config,
        include_warmstart=request.include_warmstart,
        warmstart_completed_families=set(
            request.warmstart_completed_families
        ),
        parent_pdb_available=parent_structure_available,
        structure_refilter_result_ids=structure_refilter_result_ids,
        structure_refilter_scored_source_ids=(
            structure_refilter_scored_source_ids
        ),
    ))
    feasible_candidates = tuple(
        candidate
        for candidate in candidates
        if candidate.feasibility.all_ok()
    )

    launched_candidate_ids = {
        launch.candidate_id for launch in request.prior_launches
    }
    pending_chain_candidates = tuple(
        candidate
        for candidate in request.existing_actions
        if candidate.candidate_id.startswith("chain_")
        and candidate.candidate_id not in launched_candidate_ids
        and candidate.feasibility.all_ok()
    )

    return CandidatePhaseResult(
        candidates=candidates,
        feasible_candidates=feasible_candidates,
        pending_chain_candidates=pending_chain_candidates,
        parent_structure_available=parent_structure_available,
        structure_refilter_result_ids=frozenset(structure_refilter_result_ids),
        structure_refilter_scored_source_ids=frozenset(
            structure_refilter_scored_source_ids
        ),
    )


__all__ = [
    "CandidatePhaseRequest",
    "CandidatePhaseResult",
    "build_candidate_phase",
]
