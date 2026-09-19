"""Parse and normalize scientific worker outputs before archive writes.

This module deliberately stops at the archive boundary. It converts backend
files into consistently attributed ``ResultRecord`` objects; the controller
still decides when to append those records and whether to schedule follow-up
work.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import AbstractSet, Callable, Iterable, Literal

from ..backend_extensions import (
    get_backend_adapter,
    validate_extension_records,
)
from ..output_parsers import ParseError, ParserContext
from ..output_parsers.af2_refilter import parse_af2_refilter_output
from ..output_parsers.bindcraft import parse_bindcraft_output
from ..output_parsers.boltzgen import parse_boltzgen_output
from ..output_parsers.complexa import parse_complexa_output
from ..output_parsers.proteinmpnn import parse_proteinmpnn_output
from ..refilter_roles import infer_refilter_role
from ..schemas import ActionCandidate, ResultRecord, TargetConstraint


OutputParser = Callable[[Path, ParserContext], list[ResultRecord]]


@dataclass(frozen=True)
class OutputParserRegistry:
    """Named built-in parsers used by the worker-output dispatcher."""

    bindcraft: OutputParser = parse_bindcraft_output
    af2_refilter: OutputParser = parse_af2_refilter_output
    proteinmpnn: OutputParser = parse_proteinmpnn_output
    boltzgen: OutputParser = parse_boltzgen_output
    complexa: OutputParser = parse_complexa_output


@dataclass(frozen=True)
class WorkerOutputRequest:
    """Complete, inspectable input for one worker-output processing pass."""

    candidate: ActionCandidate
    requested_output_directory: Path
    target: TargetConstraint
    tick_id: str
    parent_pdb_path: str = ""
    parent_result_id: str = ""
    target_chains_csv: str = ""
    binder_chain: str = ""
    parent_backend_family: str = ""
    existing_result_ids: frozenset[str] = field(default_factory=frozenset)
    previously_archived_result_ids: frozenset[str] = field(
        default_factory=frozenset
    )
    elapsed_gpu_hours: float = 0.0
    previously_archived_gpu_hours: float = 0.0


@dataclass(frozen=True)
class ProcessedWorkerOutput:
    """Normalized result of inspecting one worker output directory."""

    output_directory: Path
    parser_context: ParserContext
    result_records: tuple[ResultRecord, ...]
    parsed_record_count: int
    elapsed_gpu_hours_delta: float
    parser_error: str | None = None

    @property
    def duplicate_record_count(self) -> int:
        return self.parsed_record_count - len(self.result_records)


def _resolve_output_directory(
    *,
    method_family: str,
    requested_output_directory: Path,
    target_id: str,
) -> Path:
    """Find an auto-renamed Complexa directory while preserving other paths."""

    if not method_family.startswith("complexa_"):
        return requested_output_directory
    if requested_output_directory.exists():
        return requested_output_directory
    parent_directory = requested_output_directory.parent
    if not parent_directory.exists():
        return requested_output_directory
    output_suffix = requested_output_directory.name.split("_")[-1]
    matching_directories = sorted(
        parent_directory.glob(f"*{target_id}*{output_suffix}*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return matching_directories[0] if matching_directories else requested_output_directory


def _build_parser_context(
    *,
    candidate: ActionCandidate,
    target: TargetConstraint,
    tick_id: str,
    parent_pdb_path: str,
    parent_result_id: str,
    target_chains_csv: str,
    binder_chain: str,
    parent_backend_family: str,
) -> ParserContext:
    parent_ids = [candidate.candidate_id]
    if parent_result_id:
        parent_ids.append(parent_result_id)
    refilter_role = infer_refilter_role(candidate) or ""
    refilter_source_family = (
        parent_backend_family
        if candidate.method_family == "structure_refilter" and parent_result_id
        else ""
    )
    return ParserContext(
        target_id=target.target_id,
        runtime_bucket_id=candidate.feasibility.runtime_bucket_id or "rb_v7",
        candidate_id=candidate.candidate_id,
        parent_ids=parent_ids,
        method_family=candidate.method_family,
        tick_id=tick_id,
        parent_pdb_path=parent_pdb_path,
        parent_result_id=parent_result_id,
        target_chains_csv=target_chains_csv,
        binder_chain=binder_chain,
        refilter_role=refilter_role,
        refilter_source_family=refilter_source_family,
    )


def _parse_output_directory(
    *,
    method_family: str,
    output_directory: Path,
    parser_context: ParserContext,
    parser_registry: OutputParserRegistry,
) -> list[ResultRecord]:
    if method_family == "bindcraft":
        return parser_registry.bindcraft(output_directory, parser_context)
    if method_family == "structure_refilter":
        return parser_registry.af2_refilter(output_directory, parser_context)
    if method_family == "proteinmpnn_redesign":
        return parser_registry.proteinmpnn(output_directory, parser_context)
    if method_family == "boltzgen":
        return parser_registry.boltzgen(output_directory, parser_context)
    if method_family.startswith("complexa_"):
        return parser_registry.complexa(output_directory, parser_context)

    backend_adapter = get_backend_adapter(method_family)
    if backend_adapter is None:
        raise ParseError(
            f"no output parser registered for method_family {method_family!r}"
        )
    try:
        return validate_extension_records(
            backend_adapter,
            backend_adapter.parse_output(output_directory, parser_context),
            parser_context,
        )
    except (TypeError, ValueError) as error:
        raise ParseError(str(error)) from error


def _keep_new_records(
    records: Iterable[ResultRecord],
    *,
    existing_result_ids: AbstractSet[str],
    previously_archived_result_ids: AbstractSet[str],
) -> list[ResultRecord]:
    return [
        record
        for record in records
        if record.result_id not in existing_result_ids
        and record.result_id not in previously_archived_result_ids
    ]


def _normalize_records(
    records: list[ResultRecord],
    *,
    elapsed_gpu_hours_delta: float,
    binder_chain: str,
    target_chains_csv: str,
) -> list[ResultRecord]:
    normalized_records = records
    if elapsed_gpu_hours_delta > 0.0 and normalized_records:
        gpu_hours_per_record = elapsed_gpu_hours_delta / len(normalized_records)
        normalized_records = [
            replace(record, gpu_h=gpu_hours_per_record)
            for record in normalized_records
        ]

    if not normalized_records or not (binder_chain or target_chains_csv):
        return normalized_records
    annotated_records = []
    for record in normalized_records:
        bins = dict(record.bins or {})
        artifacts = dict(record.artifacts or {})
        if binder_chain:
            bins.setdefault("binder_chain", binder_chain)
            artifacts.setdefault("binder_chain", binder_chain)
        if target_chains_csv:
            bins.setdefault("target_chains", target_chains_csv)
            artifacts.setdefault("target_chains", target_chains_csv)
        annotated_records.append(replace(record, bins=bins, artifacts=artifacts))
    return annotated_records


def process_worker_output(
    request: WorkerOutputRequest,
    *,
    parser_registry: OutputParserRegistry | None = None,
) -> ProcessedWorkerOutput:
    """Parse, deduplicate, charge, and annotate one worker's results."""

    output_directory = _resolve_output_directory(
        method_family=request.candidate.method_family,
        requested_output_directory=request.requested_output_directory,
        target_id=request.target.target_id,
    )
    parser_context = _build_parser_context(
        candidate=request.candidate,
        target=request.target,
        tick_id=request.tick_id,
        parent_pdb_path=request.parent_pdb_path,
        parent_result_id=request.parent_result_id,
        target_chains_csv=request.target_chains_csv,
        binder_chain=request.binder_chain,
        parent_backend_family=request.parent_backend_family,
    )
    parser_error = None
    try:
        parsed_records = _parse_output_directory(
            method_family=request.candidate.method_family,
            output_directory=output_directory,
            parser_context=parser_context,
            parser_registry=parser_registry or OutputParserRegistry(),
        )
    except ParseError as error:
        parsed_records = []
        parser_error = str(error)
    parsed_record_count = len(parsed_records)
    new_records = _keep_new_records(
        parsed_records,
        existing_result_ids=request.existing_result_ids,
        previously_archived_result_ids=request.previously_archived_result_ids,
    )
    elapsed_gpu_hours_delta = max(
        0.0,
        request.elapsed_gpu_hours - request.previously_archived_gpu_hours,
    )
    normalized_records = _normalize_records(
        new_records,
        elapsed_gpu_hours_delta=elapsed_gpu_hours_delta,
        binder_chain=request.binder_chain,
        target_chains_csv=request.target_chains_csv,
    )
    return ProcessedWorkerOutput(
        output_directory=output_directory,
        parser_context=parser_context,
        result_records=tuple(normalized_records),
        parsed_record_count=parsed_record_count,
        elapsed_gpu_hours_delta=elapsed_gpu_hours_delta,
        parser_error=parser_error,
    )


def synthetic_result_exit_status(
    return_code: int,
) -> Literal["timeout", "nonzero_exit", "no_artifacts"]:
    """Map a worker return code to the synthetic-result lifecycle vocabulary."""

    if return_code == -1:
        return "timeout"
    if return_code != 0:
        return "nonzero_exit"
    return "no_artifacts"


def create_synthetic_result_record(
    *,
    candidate: ActionCandidate,
    target_id: str,
    parent_ids: list[str],
    tick_id: str,
    return_code: int,
    elapsed_gpu_hours: float,
    refilter_role: str,
    refilter_source_family: str,
    completion_after_incremental_parse: bool = False,
    created_at_epoch_seconds: float | None = None,
) -> ResultRecord:
    """Create an auditable no-artifact result for consumed GPU time."""

    timestamp = (
        time.time()
        if created_at_epoch_seconds is None
        else created_at_epoch_seconds
    )
    completion_label = "completion_" if completion_after_incremental_parse else ""
    bins = {
        **({"refilter_role": refilter_role} if refilter_role else {}),
        **(
            {"refilter_source_family": refilter_source_family}
            if refilter_source_family else {}
        ),
        "return_code": str(return_code),
    }
    if completion_after_incremental_parse:
        bins["completion_after_incremental_parse"] = "1"
    return ResultRecord(
        result_id=(
            f"synth_{candidate.method_family}_{candidate.candidate_id}_"
            f"{completion_label}{int(timestamp):x}"
        )[:80],
        parent_ids=list(parent_ids),
        target_id=target_id,
        backend_family=candidate.method_family,
        runtime_bucket_id=candidate.feasibility.runtime_bucket_id or "rb_v7",
        metrics={},
        metrics_calibrated={},
        route_lineage=[candidate.method_family],
        gpu_h=elapsed_gpu_hours,
        exit_status=synthetic_result_exit_status(return_code),
        bins=bins,
        artifacts={},
        panel_ready=False,
        tick_id=tick_id or None,
    )


__all__ = [
    "OutputParser",
    "OutputParserRegistry",
    "ProcessedWorkerOutput",
    "WorkerOutputRequest",
    "create_synthetic_result_record",
    "process_worker_output",
    "synthetic_result_exit_status",
]
