"""Create the fail-closed provenance record for a campaign run."""

from __future__ import annotations

import argparse
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from ..provenance import capture_run_provenance
from .artifacts import CampaignArtifacts
from .models import ResolvedCampaign


@dataclass(frozen=True)
class CampaignProvenanceArtifact:
    """Written provenance path and identities needed by the controller."""

    path: Path
    source_tree_sha256: str
    model_content_sha256: str


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _next_provenance_path(archive_root: Path) -> Path:
    primary = archive_root / "run_provenance.json"
    if not primary.exists():
        return primary
    sequence = 1
    while True:
        candidate = archive_root / f"run_provenance_resume_{sequence:03d}.json"
        if not candidate.exists():
            return candidate
        sequence += 1


def write_campaign_provenance(
    campaign: ResolvedCampaign,
    campaign_artifacts: CampaignArtifacts,
) -> CampaignProvenanceArtifact:
    """Write complete source/model/target provenance before worker startup.

    A real campaign run requires a locally content-addressed model.  Development
    against an unverified remote endpoint remains available through the legacy
    launcher, but cannot accidentally use this publication-oriented interface.
    """

    runtime_paths = campaign.runtime_paths()
    model_path = runtime_paths.qwen_model_path
    model_manifest = runtime_paths.qwen_model_manifest
    if model_path is None or model_manifest is None:
        raise ValueError(
            "campaign run requires backends.qwen_model_path and "
            "backends.qwen_model_manifest for run provenance"
        )

    executed_source_root = Path(__file__).resolve().parents[2]
    if runtime_paths.repo_root.resolve() != executed_source_root:
        raise ValueError(
            "backends.repo_root must identify the T-REX source executing this "
            f"campaign: running={executed_source_root} "
            f"configured={runtime_paths.repo_root.resolve()}"
        )

    output_path = _next_provenance_path(campaign.config.run.archive_root)
    policy = campaign.config.policy
    client_model = campaign.config.llm.model
    served_model = (
        client_model.removeprefix("vllm/")
        if client_model.startswith("vllm/")
        else client_model
    )
    args = argparse.Namespace(
        out=str(output_path),
        source_root=str(executed_source_root),
        model_path=str(model_path),
        model_manifest=str(model_manifest),
        served_model=served_model,
        llm_model=client_model,
        llm_base_url=campaign.config.llm.base_url,
        target=campaign.target.name,
        target_id=campaign.target.target_id,
        target_pdb=str(campaign.target.pdb_path),
        target_config=str(campaign.target.config_path),
        max_wall_h=str(campaign.config.run.max_wall_hours),
        foldseek_su_tm_score=str(policy.foldseek_su_tm_score),
        foldseek_collapse_tm_score=str(policy.foldseek_collapse_tm_score),
        enabled_families=",".join(campaign.effective_families),
        worker_gpus=",".join(campaign.config.run.worker_gpus),
        charged_gpus=os.environ.get("TREX_CHARGED_GPUS"),
        seed=str(campaign.config.run.seed),
        critic_enabled=str(int(policy.critic)),
        evidence_skip_enabled=str(int(policy.evidence_skip)),
        campaign_name=campaign.config.name,
        campaign_source_path=str(campaign.source_path),
        campaign_source_sha256=campaign.source_sha256,
        campaign_input_artifact=str(campaign_artifacts.input_path),
        campaign_input_artifact_sha256=_file_sha256(campaign_artifacts.input_path),
        campaign_resolved_artifact=str(campaign_artifacts.resolved_path),
        campaign_resolved_artifact_sha256=_file_sha256(
            campaign_artifacts.resolved_path
        ),
    )
    payload = capture_run_provenance(args)
    return CampaignProvenanceArtifact(
        path=output_path,
        source_tree_sha256=payload["source"]["tree_sha256"],
        model_content_sha256=payload["model"]["content_sha256"],
    )
