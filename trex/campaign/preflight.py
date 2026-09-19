"""Preflight the resolved campaign contract without starting workers."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ..cross_campaign_memory import cross_campaign_memory_metadata
from ..validation import Check, validate_install
from .environment import applied_environment
from .models import ResolvedCampaign


PREFLIGHT_REPORT_SCHEMA_VERSION = "trex.preflight-report.v1"


@dataclass(frozen=True)
class PreflightReport:
    """Structured checks suitable for both terminal and JSON output."""

    checks: tuple[Check, ...]

    @property
    def failures(self) -> tuple[Check, ...]:
        return tuple(
            check for check in self.checks if check.required and check.status == "fail"
        )

    @property
    def ok(self) -> bool:
        return not self.failures

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PREFLIGHT_REPORT_SCHEMA_VERSION,
            "ok": self.ok,
            "failure_count": len(self.failures),
            "checks": [asdict(check) for check in self.checks],
        }


def _nearest_existing_parent(path: Path) -> Path:
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def _input_checks(campaign: ResolvedCampaign) -> list[Check]:
    parsed = urlparse(campaign.config.llm.base_url)
    endpoint_ok = parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    archive = campaign.config.run.archive_root
    parent = _nearest_existing_parent(archive)
    executed_source_root = Path(__file__).resolve().parents[2]
    configured_source_root = campaign.runtime_paths().repo_root.resolve()
    writable = parent.is_dir() and os.access(parent, os.W_OK)
    archive_detail = (
        f"{archive} (existing archive; controller will resume)"
        if archive.is_dir()
        else f"{archive} (will be created; nearest parent: {parent})"
    )
    checks = [
        Check(
            "campaign source SHA256",
            "ok",
            campaign.source_sha256,
        ),
        Check(
            "executed T-ReX source",
            "ok" if configured_source_root == executed_source_root else "fail",
            (
                str(executed_source_root)
                if configured_source_root == executed_source_root
                else (
                    f"running={executed_source_root} "
                    f"configured={configured_source_root}"
                )
            ),
        ),
        Check(
            "archive output",
            "ok" if writable else "fail",
            archive_detail,
        ),
        Check(
            "worker GPU mapping",
            "ok",
            ",".join(campaign.config.run.worker_gpus),
        ),
        Check(
            "LLM endpoint",
            "ok" if endpoint_ok else "fail",
            campaign.config.llm.base_url,
        ),
    ]
    memory_path = campaign.config.memory.cross_campaign_path
    if memory_path is None:
        checks.append(Check("cross-campaign memory", "ok", "disabled", required=False))
    else:
        try:
            metadata = cross_campaign_memory_metadata(memory_path)
        except (OSError, ValueError, TypeError) as exc:
            checks.append(
                Check(
                    "cross-campaign memory",
                    "fail",
                    f"{type(exc).__name__}: {exc}",
                )
            )
        else:
            checks.append(
                Check(
                    "cross-campaign memory",
                    "ok",
                    f"{memory_path} sha256={metadata['sha256']}",
                )
            )
    return checks


def preflight_campaign(
    campaign: ResolvedCampaign,
    *,
    require_backends: bool = False,
    require_model: bool = False,
    verify_asset_hash: bool = True,
    verify_backend_revisions: bool = False,
    verify_checkpoint_content: bool = False,
) -> PreflightReport:
    """Validate user input, target assets, adapters, and optional checkpoints."""

    with applied_environment(campaign.environment()):
        installation = validate_install(
            target=campaign.target.name,
            enabled_families=campaign.effective_families,
            target_config=campaign.target.config_path,
            target_pdb=campaign.target.pdb_path,
            asset_root=campaign.config.target.asset_root,
            require_backends=require_backends,
            require_model=require_model,
            verify_asset_hash=verify_asset_hash,
            verify_backend_revisions=verify_backend_revisions,
            verify_checkpoint_content=verify_checkpoint_content,
        )
    return PreflightReport(tuple(_input_checks(campaign) + installation))
