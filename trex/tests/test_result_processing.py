"""Contracts for backend-independent worker result processing."""

from __future__ import annotations

from pathlib import Path

import pytest

from trex.execution.result_processing import (
    OutputParser,
    OutputParserRegistry,
    WorkerOutputRequest,
    create_synthetic_result_record,
    process_worker_output,
)
from trex.output_parsers import ParseError, ParserContext
from trex.schemas import (
    ActionCandidate,
    FeasibilityCheck,
    ResultRecord,
    TargetConstraint,
)


def _candidate(
    method_family: str = "bindcraft",
    *,
    candidate_id: str = "candidate-1",
) -> ActionCandidate:
    return ActionCandidate(
        candidate_id=candidate_id,
        hypothesis_ids=["hypothesis-1"],
        parent_result_id="parent-1",
        method_family=method_family,
        operator_id="operator-1",
        lane_id="lane-1",
        config_delta={},
        downstream_route_plan=[],
        estimated_cost_class="standard",  # type: ignore[arg-type]
        expected_signal="test",
        evidence_refs=["parent-1"],
        feasibility=FeasibilityCheck(
            backend_healthy=True,
            runtime_bucket_id="runtime-1",
            compiler_ok=True,
            verifier_ok=True,
            route_cap_ok=True,
            cost_ok=True,
        ),
    )


def _result(result_id: str, method_family: str = "bindcraft") -> ResultRecord:
    return ResultRecord(
        result_id=result_id,
        parent_ids=["candidate-1"],
        target_id="target-1",
        backend_family=method_family,
        runtime_bucket_id="runtime-1",
        metrics={},
        metrics_calibrated={},
        route_lineage=[method_family],
        gpu_h=99.0,
        exit_status="ok",
    )


def _empty_parser(
    output_directory: Path,
    parser_context: ParserContext,
) -> list[ResultRecord]:
    del output_directory, parser_context
    return []


def _registry(**overrides: OutputParser) -> OutputParserRegistry:
    parsers = {
        "bindcraft": _empty_parser,
        "af2_refilter": _empty_parser,
        "proteinmpnn": _empty_parser,
        "boltzgen": _empty_parser,
        "complexa": _empty_parser,
    }
    parsers.update(overrides)
    return OutputParserRegistry(**parsers)


def test_worker_output_processing_deduplicates_charges_and_annotates(
    tmp_path: Path,
) -> None:
    parsed_records = [
        _result("already-in-archive"),
        _result("already-from-this-worker"),
        _result("new-result"),
    ]
    observed_contexts = []

    def parse_bindcraft(output_directory, parser_context):
        assert output_directory == tmp_path
        observed_contexts.append(parser_context)
        return parsed_records

    request = WorkerOutputRequest(
        candidate=_candidate(),
        requested_output_directory=tmp_path,
        target=TargetConstraint(target_id="target-1", target_class="test"),
        tick_id="tick-7",
        parent_pdb_path="/inputs/parent.pdb",
        parent_result_id="parent-1",
        target_chains_csv="A,C",
        binder_chain="D",
        parent_backend_family="complexa_beam",
        existing_result_ids=frozenset({"already-in-archive"}),
        previously_archived_result_ids=frozenset(
            {"already-from-this-worker"}
        ),
        elapsed_gpu_hours=2.0,
        previously_archived_gpu_hours=0.5,
    )
    processed = process_worker_output(
        request,
        parser_registry=_registry(bindcraft=parse_bindcraft),
    )

    assert processed.parsed_record_count == 3
    assert processed.duplicate_record_count == 2
    assert processed.elapsed_gpu_hours_delta == pytest.approx(1.5)
    assert [record.result_id for record in processed.result_records] == [
        "new-result"
    ]
    result = processed.result_records[0]
    assert result.gpu_h == pytest.approx(1.5)
    assert result.bins["binder_chain"] == "D"
    assert result.bins["target_chains"] == "A,C"
    assert result.artifacts["binder_chain"] == "D"
    assert observed_contexts[0].parent_ids == ["candidate-1", "parent-1"]


def test_refilter_context_records_parent_backend_family(tmp_path: Path) -> None:
    observed_contexts = []

    def parse_af2_refilter(output_directory, parser_context):
        observed_contexts.append(parser_context)
        return []

    request = WorkerOutputRequest(
        candidate=_candidate("structure_refilter"),
        requested_output_directory=tmp_path,
        target=TargetConstraint(target_id="target-1", target_class="test"),
        tick_id="tick-8",
        parent_pdb_path="/inputs/parent.pdb",
        parent_result_id="parent-1",
        target_chains_csv="A",
        binder_chain="B",
        parent_backend_family="boltzgen",
    )
    process_worker_output(
        request,
        parser_registry=_registry(af2_refilter=parse_af2_refilter),
    )

    assert observed_contexts[0].refilter_source_family == "boltzgen"


def test_complexa_auto_renamed_output_directory_is_resolved(tmp_path: Path) -> None:
    requested_directory = tmp_path / "expected_007"
    actual_directory = tmp_path / "worker_target-1_007_results"
    actual_directory.mkdir()
    observed_directories = []

    def parse_complexa(output_directory, parser_context):
        observed_directories.append(output_directory)
        return []

    request = WorkerOutputRequest(
        candidate=_candidate("complexa_beam"),
        requested_output_directory=requested_directory,
        target=TargetConstraint(target_id="target-1", target_class="test"),
        tick_id="tick-9",
        target_chains_csv="A",
        binder_chain="B",
    )
    processed = process_worker_output(
        request,
        parser_registry=_registry(complexa=parse_complexa),
    )

    assert processed.output_directory == actual_directory
    assert observed_directories == [actual_directory]


def test_expected_parser_error_becomes_inspectable_result(tmp_path: Path) -> None:
    def failing_parser(output_directory, parser_context):
        raise ParseError("malformed output")

    request = WorkerOutputRequest(
        candidate=_candidate(),
        requested_output_directory=tmp_path,
        target=TargetConstraint(target_id="target-1", target_class="test"),
        tick_id="tick-10",
        elapsed_gpu_hours=1.0,
    )
    processed = process_worker_output(
        request,
        parser_registry=_registry(bindcraft=failing_parser),
    )

    assert processed.result_records == ()
    assert processed.parser_error == "malformed output"
    assert processed.elapsed_gpu_hours_delta == pytest.approx(1.0)


@pytest.mark.parametrize(
    "return_code, expected_status",
    [(-1, "timeout"), (-11, "nonzero_exit"), (0, "no_artifacts")],
)
def test_synthetic_result_uses_explicit_lifecycle_status(
    return_code: int,
    expected_status: str,
) -> None:
    synthetic_result = create_synthetic_result_record(
        candidate=_candidate(),
        target_id="target-1",
        parent_ids=["candidate-1", "parent-1"],
        tick_id="tick-11",
        return_code=return_code,
        elapsed_gpu_hours=0.25,
        refilter_role="canonical_score_conversion",
        refilter_source_family="boltzgen",
        created_at_epoch_seconds=1234.0,
    )

    assert synthetic_result.exit_status == expected_status
    assert synthetic_result.gpu_h == pytest.approx(0.25)
    assert synthetic_result.bins["return_code"] == str(return_code)
    assert synthetic_result.result_id.endswith("4d2")
