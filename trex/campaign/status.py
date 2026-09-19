"""Read-only, compact status view over an append-only campaign archive."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..archive import Archive
from ..dispatch_outcomes import count_dispatch_outcomes
from ..schemas import DispatchRecord, EvidenceSummary, LaunchDecision, ResultRecord


STATUS_SCHEMA_VERSION = "trex.campaign-status.v2"


@dataclass(frozen=True)
class CampaignStatus:
    archive_root: Path
    campaign_input: dict[str, Any] | None
    input_artifact_status: str
    record_counts: dict[str, int]
    dispatch_counts: dict[str, int]
    dispatch_outcome_counts: dict[str, int]
    launch_counts: dict[str, int]
    result_counts: dict[str, int]
    latest_evidence: dict[str, Any] | None
    provenance_files: tuple[str, ...]
    skipped_records: dict[str, int]
    warnings: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": STATUS_SCHEMA_VERSION,
            "archive_root": str(self.archive_root),
            "campaign": self.campaign_input,
            "input_artifact_status": self.input_artifact_status,
            "record_counts": self.record_counts,
            "execution": {
                "dispatch_status": self.dispatch_counts,
                "dispatch_outcome": self.dispatch_outcome_counts,
                "launch_status": self.launch_counts,
                "result_exit_status": self.result_counts,
            },
            "latest_evidence": self.latest_evidence,
            "provenance": {
                "present": bool(self.provenance_files),
                "files": list(self.provenance_files),
            },
            "integrity": {
                "scope": "typed read health; run trex-analyze validate for full validation",
                "skipped_records": self.skipped_records,
                "warnings": list(self.warnings),
            },
        }


def _latest_resolved_input(
    root: Path,
) -> tuple[dict[str, Any] | None, str | None]:
    candidates = sorted(
        root.glob("campaign_resolved*.json"),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )
    if not candidates:
        return None, None
    try:
        payload = json.loads(candidates[-1].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"{candidates[-1].name}: invalid resolved input: {exc}"
    if not isinstance(payload, dict):
        return None, f"{candidates[-1].name}: resolved input must be a JSON object"
    sections: dict[str, dict[str, Any]] = {}
    for section_name in ("campaign", "target", "run"):
        section = payload.get(section_name)
        if section is None:
            sections[section_name] = {}
        elif isinstance(section, dict):
            sections[section_name] = section
        else:
            return None, (
                f"{candidates[-1].name}: resolved input section "
                f"{section_name!r} must be a JSON object"
            )
    campaign = sections["campaign"]
    target = sections["target"]
    run = sections["run"]
    return {
        "name": campaign.get("name"),
        "target": target.get("name"),
        "target_id": target.get("target_id"),
        "source_sha256": campaign.get("source_sha256"),
        "max_wall_hours": run.get("max_wall_hours"),
        "worker_gpus": run.get("worker_gpus"),
        "enabled_families": run.get("enabled_families"),
    }, None


def _compact_evidence(evidence: EvidenceSummary | None) -> dict[str, Any] | None:
    if evidence is None:
        return None
    names = (
        "tick_id",
        "target_id",
        "state_label",
        "completed_children",
        "pending_children",
        "strict_count",
        "run_su_count",
        "elapsed_wall_h",
        "remaining_wall_h",
        "worker_gpu_h_total",
        "worker_wall_gpu_count",
        "worker_wall_gpu_h_total",
        "run_su_per_worker_wall_gpu_h_total",
        "gpu_h_since_last_su",
        "production_panel_status",
        "production_panel_value",
    )
    return {
        name: getattr(evidence, name)
        for name in names
        if hasattr(evidence, name) and getattr(evidence, name) is not None
    }


def _provenance_input_summary(root: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Recover a compact input view for provenance-only Slurm archives."""

    candidates = sorted(
        root.glob("run_provenance*.json"),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )
    if not candidates:
        return None, None
    path = candidates[-1]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"{path.name}: invalid provenance input: {exc}"
    if not isinstance(payload, dict):
        return None, f"{path.name}: provenance input must be a JSON object"
    target = payload.get("target") if isinstance(payload.get("target"), dict) else {}
    controller = (
        payload.get("controller") if isinstance(payload.get("controller"), dict) else {}
    )
    runtime = payload.get("runtime") if isinstance(payload.get("runtime"), dict) else {}
    return {
        "name": runtime.get("slurm_job_name"),
        "target": target.get("name"),
        "target_id": target.get("target_id"),
        "source_sha256": None,
        "max_wall_hours": controller.get("max_wall_h"),
        "worker_gpus": controller.get("worker_gpus"),
        "enabled_families": controller.get("enabled_families"),
    }, None


def collect_campaign_status(archive_root: Path | str) -> CampaignStatus:
    """Collect status without creating or modifying the requested archive."""

    root = Path(archive_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"campaign archive does not exist: {root}")
    archive = Archive(root)
    record_counts = archive.summary()
    dispatch_records = tuple(archive.iter_records(DispatchRecord))
    dispatch_counts = Counter(record.status for record in dispatch_records)
    launch_counts = Counter(
        record.status for record in archive.iter_records(LaunchDecision)
    )
    result_counts = Counter(
        record.exit_status for record in archive.iter_records(ResultRecord)
    )
    evidence = archive.latest_evidence()
    campaign_input, campaign_input_warning = _latest_resolved_input(root)
    provenance_files = tuple(
        path.name for path in sorted(root.glob("run_provenance*.json"))
    )
    skipped_records = archive.read_skip_counts()
    warnings: list[str] = []
    if campaign_input_warning is not None:
        input_artifact_status = "invalid_campaign_resolved"
        warnings.append(campaign_input_warning)
    elif campaign_input is None:
        provenance_input, provenance_input_warning = _provenance_input_summary(root)
        if provenance_input_warning is not None:
            input_artifact_status = "invalid_provenance"
            warnings.append(provenance_input_warning)
        elif provenance_input is not None:
            input_artifact_status = "provenance_only"
            campaign_input = provenance_input
        else:
            input_artifact_status = "missing"
            warnings.append("no campaign_resolved*.json input artifact")
    else:
        input_artifact_status = "campaign_resolved"
    if not provenance_files:
        warnings.append("no run_provenance*.json reproducibility artifact")
    if skipped_records:
        warnings.append("one or more malformed/truncated archive records were skipped")
    return CampaignStatus(
        archive_root=root,
        campaign_input=campaign_input,
        input_artifact_status=input_artifact_status,
        record_counts=record_counts,
        dispatch_counts=dict(sorted(dispatch_counts.items())),
        dispatch_outcome_counts=count_dispatch_outcomes(dispatch_records),
        launch_counts=dict(sorted(launch_counts.items())),
        result_counts=dict(sorted(result_counts.items())),
        latest_evidence=_compact_evidence(evidence),
        provenance_files=provenance_files,
        skipped_records=skipped_records,
        warnings=tuple(warnings),
    )
