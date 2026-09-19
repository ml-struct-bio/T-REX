"""Human-readable rendering for campaign inputs and outputs."""

from __future__ import annotations

from typing import Any, Iterable

from .models import ResolvedCampaign
from .preflight import PreflightReport
from .status import CampaignStatus


def _section(title: str, rows: Iterable[tuple[str, Any]]) -> list[str]:
    materialized = [(key, value) for key, value in rows]
    width = max((len(key) for key, _ in materialized), default=0)
    lines = [title]
    lines.extend(f"  {key:<{width}}  {value}" for key, value in materialized)
    return lines


def render_campaign(campaign: ResolvedCampaign) -> str:
    config = campaign.config
    runtime_paths = campaign.runtime_paths()
    sections = [
        _section(
            "Campaign",
            (
                ("name", config.name),
                ("source", campaign.source_path),
                ("source SHA256", campaign.source_sha256),
            ),
        ),
        _section(
            "Target input",
            (
                ("name", campaign.target.name),
                ("target ID", campaign.target.target_id),
                ("constraint", campaign.target.config_path),
                ("PDB", campaign.target.pdb_path),
            ),
        ),
        _section(
            "Execution",
            (
                ("archive", config.run.archive_root),
                ("controller time limit", f"{config.run.max_wall_hours:g} h (cumulative)"),
                ("seed", config.run.seed),
                ("worker GPUs", ", ".join(config.run.worker_gpus)),
                ("families", ", ".join(campaign.effective_families)),
            ),
        ),
        _section(
            "Decision policy",
            (
                ("LLM endpoint", config.llm.base_url),
                ("LLM model", config.llm.model),
                ("critic", "on" if config.policy.critic else "off"),
                ("evidence skip", "on" if config.policy.evidence_skip else "off"),
                ("exemplars", "on" if config.policy.exemplars else "off"),
                ("Foldseek SU TM", config.policy.foldseek_su_tm_score),
                ("collapse TM", config.policy.foldseek_collapse_tm_score),
                ("quota realization", config.policy.selector_quota_realization),
                (
                    "cross-campaign memory",
                    config.memory.cross_campaign_path or "disabled",
                ),
            ),
        ),
        _section(
            "Backend runtime",
            (
                ("T-ReX repository", runtime_paths.repo_root),
                ("external root", runtime_paths.external_root),
                ("Complexa repository", runtime_paths.complexa_repo),
                ("Complexa Python", runtime_paths.complexa_python),
                ("legacy Complexa repository", runtime_paths.legacy_complexa_repo),
                ("AF2 parameters", runtime_paths.af2_data_dir),
                ("ProteinMPNN weights", runtime_paths.proteinmpnn_weights),
                ("BindCraft repository", runtime_paths.bindcraft_repo),
                ("BindCraft environment", runtime_paths.bindcraft_env),
                ("BoltzGen repository", runtime_paths.boltzgen_repo),
                ("BoltzGen binary", runtime_paths.boltzgen_binary),
                ("BoltzGen cache", runtime_paths.boltzgen_cache),
                ("Foldseek binary", runtime_paths.foldseek_command),
                ("MMseqs binary", runtime_paths.mmseqs_command),
                (
                    "Qwen model path",
                    runtime_paths.qwen_model_path or "not configured",
                ),
                (
                    "Qwen model manifest",
                    runtime_paths.qwen_model_manifest or "not configured",
                ),
            ),
        ),
    ]
    if campaign.resolution_notes:
        sections.append(
            _section(
                "Resolution notes",
                (
                    (str(index), note)
                    for index, note in enumerate(campaign.resolution_notes, 1)
                ),
            )
        )
    return "\n\n".join("\n".join(section) for section in sections)


def render_preflight(report: PreflightReport) -> str:
    width = max((len(check.name) for check in report.checks), default=0)
    lines = [
        f"{check.status.upper():4}  {check.name:<{width}}  {check.detail}"
        for check in report.checks
    ]
    lines.append(
        "\nPreflight passed."
        if report.ok
        else f"\nPreflight failed: {len(report.failures)} required check(s)."
    )
    return "\n".join(lines)


def render_status(status: CampaignStatus) -> str:
    sections: list[list[str]] = []
    identity = status.campaign_input or {"name": "not recorded"}
    sections.append(
        _section(
            "Campaign output",
            (
                ("archive", status.archive_root),
                ("name", identity.get("name")),
                ("target", identity.get("target")),
                ("target ID", identity.get("target_id")),
                ("input artifact", status.input_artifact_status),
            ),
        )
    )
    sections.append(
        _section(
            "Record streams",
            ((name, count) for name, count in status.record_counts.items() if count),
        )
    )
    sections.append(
        _section(
            "Execution",
            (
                ("launches", status.launch_counts or "none"),
                ("dispatch outcomes", status.dispatch_outcome_counts or "none"),
                ("raw dispatch status", status.dispatch_counts or "none"),
                ("results", status.result_counts or "none"),
            ),
        )
    )
    if status.latest_evidence:
        sections.append(_section("Latest evidence", status.latest_evidence.items()))
    sections.append(
        _section(
            "Reproducibility",
            (("provenance", ", ".join(status.provenance_files) or "MISSING"),),
        )
    )
    sections.append(
        _section(
            "Read integrity",
            (
                (
                    "scope",
                    "typed read health; use trex-analyze validate for full validation",
                ),
                ("skipped records", status.skipped_records or "none"),
                ("warnings", "; ".join(status.warnings) or "none"),
            ),
        )
    )
    return "\n\n".join("\n".join(section) for section in sections)
