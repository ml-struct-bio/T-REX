"""Independent contracts for evidence-driven backend-family disabling."""

from __future__ import annotations

from types import SimpleNamespace

from trex.execution.circuit_breaker import (
    FamilyCircuitBreakerDependencies,
    FamilyCircuitBreakerRequest,
    evaluate_family_circuit_breakers,
)
from trex.schemas import ActionCandidate, FeasibilityCheck, ResultRecord


def _candidate(candidate_id: str, method_family: str) -> ActionCandidate:
    return ActionCandidate(
        candidate_id=candidate_id,
        hypothesis_ids=["hypothesis-1"],
        parent_result_id=None,
        method_family=method_family,
        operator_id=f"{method_family}_default",
        lane_id=method_family,
        config_delta={},
        downstream_route_plan=[],
        estimated_cost_class="standard",  # type: ignore[arg-type]
        expected_signal="Generate a binder candidate.",
        evidence_refs=["evidence-1"],
        feasibility=FeasibilityCheck(
            backend_healthy=True,
            runtime_bucket_id="runtime-v1",
            compiler_ok=True,
            verifier_ok=True,
            route_cap_ok=True,
            cost_ok=True,
        ),
    )


class FakeArchive:
    def __init__(self, candidates: list[ActionCandidate]) -> None:
        self.candidates = candidates
        self.scan_counts: dict[type, int] = {}

    def iter_records(self, record_type: type):
        self.scan_counts[record_type] = self.scan_counts.get(record_type, 0) + 1
        if record_type is ActionCandidate:
            return iter(self.candidates)
        if record_type is ResultRecord:
            return iter(())
        raise AssertionError(f"unexpected archive record type: {record_type}")


def test_breaker_returns_a_state_delta_and_filters_blocked_pending_work() -> None:
    archive = FakeArchive([
        _candidate("bad-1", "bad_generator"),
        _candidate("good-1", "good_generator"),
    ])
    evaluated_families: list[str] = []
    index_calls = 0

    def index_actions(candidates, results):
        nonlocal index_calls
        index_calls += 1
        assert len(candidates) == 2
        assert results == []
        return {}

    def is_broken(archive_arg, family, **kwargs):
        assert archive_arg is archive
        assert kwargs["results"] == []
        evaluated_families.append(family)
        return family == "bad_generator"

    result = evaluate_family_circuit_breakers(
        FamilyCircuitBreakerRequest(
            archive=archive,
            all_families=("bad_generator", "good_generator"),
            unavailable_families=(),
            already_blocked_families=frozenset(),
            pending_candidate_ids=("bad-1", "good-1"),
            backend_registry={
                family: SimpleNamespace(
                    role="generator", requires_parent_pdb=False
                )
                for family in ("bad_generator", "good_generator")
            },
            max_no_yield_timeouts=2,
            minimum_gpu_hours=3.0,
        ),
        FamilyCircuitBreakerDependencies(
            index_actions_by_spawned_result=index_actions,
            is_family_circuit_broken=is_broken,
        ),
    )

    assert result.newly_blocked_families == ("bad_generator",)
    assert result.blocked_families == frozenset({"bad_generator"})
    assert result.pending_candidate_ids == ("good-1",)
    assert evaluated_families == ["bad_generator", "good_generator"]
    assert index_calls == 1
    assert archive.scan_counts[ResultRecord] == 1


def test_breaker_preserves_the_only_route_light_generator() -> None:
    archive = FakeArchive([])
    is_broken_calls = 0

    def is_broken(*args, **kwargs):
        nonlocal is_broken_calls
        is_broken_calls += 1
        return True

    result = evaluate_family_circuit_breakers(
        FamilyCircuitBreakerRequest(
            archive=archive,
            all_families=("only_generator", "structure_refilter"),
            unavailable_families=(),
            already_blocked_families=frozenset(),
            pending_candidate_ids=(),
            backend_registry={
                "only_generator": SimpleNamespace(
                    role="generator", requires_parent_pdb=False
                ),
            },
            max_no_yield_timeouts=1,
            minimum_gpu_hours=0.0,
        ),
        FamilyCircuitBreakerDependencies(
            index_actions_by_spawned_result=lambda candidates, results: {},
            is_family_circuit_broken=is_broken,
        ),
    )

    assert result.newly_blocked_families == ()
    assert is_broken_calls == 0
