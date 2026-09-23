"""Typed campaign input and resolved execution models.

These dataclasses deliberately contain no parsing or terminal-rendering code.
They are the stable hand-off between the user input layer and the existing
scientific controller.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..targets import ResolvedTarget
from ..validation import DEFAULT_FAMILIES
from .environment import inspectable_environment
from .runtime.paths import (
    BACKEND_PATH_ENVIRONMENT_VARIABLES,
    RuntimePaths,
)


CAMPAIGN_SCHEMA_VERSION = "trex.campaign.v1"
RESOLVED_SCHEMA_VERSION = "trex.resolved-campaign.v1"


@dataclass(frozen=True)
class TargetInput:
    """Target name plus either registered assets or explicit custom files."""

    name: str
    asset_root: Path | None = None
    constraint: Path | None = None
    pdb: Path | None = None


@dataclass(frozen=True)
class RunConfig:
    """Cumulative controller time limit and execution settings.

    This is not the scheduler allocation or the final-analysis denominator.
    Explicit values in an existing campaign file are preserved on resume.
    """

    archive_root: Path
    max_wall_hours: float = 48.0
    seed: int = 0
    worker_gpus: tuple[str, ...] = ("1", "2", "3")
    enabled_families: tuple[str, ...] = DEFAULT_FAMILIES


@dataclass(frozen=True)
class LLMConfig:
    """Planner and Supervisor serving endpoint."""

    base_url: str = "http://127.0.0.1:12000/v1"
    model: str = "vllm/Qwen/Qwen3.6-27B-FP8"


@dataclass(frozen=True)
class PolicyConfig:
    """Named controller policies that are safe for users to configure."""

    critic: bool = True
    evidence_skip: bool = False
    exemplars: bool = True
    foldseek_su_tm_score: float = 0.60
    foldseek_collapse_tm_score: float = 0.60
    selector_quota_realization: str = "fractional_carry"
    selector_mode_window_k: int = 1
    selector_adaptive_mode_window_k: bool = False


@dataclass(frozen=True)
class MemoryConfig:
    """Optional, content-addressed historical evidence for the LLMs."""

    cross_campaign_path: Path | None = None


@dataclass(frozen=True)
class BackendPaths:
    """Explicit backend and model paths supplied through campaign configuration or TREX_* variables."""

    repo_root: Path | None = None
    external_root: Path | None = None
    complexa_repo: Path | None = None
    legacy_complexa_repo: Path | None = None
    complexa_python: Path | None = None
    bindcraft_repo: Path | None = None
    bindcraft_env: Path | None = None
    boltzgen_repo: Path | None = None
    boltzgen_binary: Path | None = None
    boltzgen_cache: Path | None = None
    foldseek_binary: Path | None = None
    mmseqs_binary: Path | None = None
    qwen_model_path: Path | None = None
    qwen_model_manifest: Path | None = None

    def environment(self) -> dict[str, str]:
        """Return the compatibility environment consumed by current adapters."""

        environment: dict[str, str] = {}
        for field_name, environment_name in (
            BACKEND_PATH_ENVIRONMENT_VARIABLES.items()
        ):
            path = getattr(self, field_name)
            if path is not None:
                environment[environment_name] = str(path)
        return environment

    def as_dict(self) -> dict[str, str | None]:
        return {
            name: str(value) if value is not None else None
            for name, value in vars(self).items()
        }

    def to_runtime_paths(self) -> RuntimePaths:
        """Materialize the immutable controller view of these backend paths."""

        return RuntimePaths.from_environment(
            self.environment(),
            default_repo_root=self.repo_root,
        )


@dataclass(frozen=True)
class CampaignConfig:
    """Validated, but not yet target-resolved, campaign input."""

    name: str
    target: TargetInput
    run: RunConfig
    llm: LLMConfig = field(default_factory=LLMConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    backends: BackendPaths = field(default_factory=BackendPaths)


@dataclass(frozen=True)
class ResolvedCampaign:
    """Concrete execution input with absolute paths and effective families."""

    config: CampaignConfig
    target: ResolvedTarget
    source_path: Path
    source_content: bytes = field(repr=False, compare=False)
    source_sha256: str
    effective_families: tuple[str, ...]
    resolution_notes: tuple[str, ...] = ()

    def controller_argv(self) -> list[str]:
        """Translate the typed contract to the legacy controller CLI boundary."""

        run = self.config.run
        policy = self.config.policy
        return [
            "--archive-root", str(run.archive_root),
            "--target-constraint", str(self.target.config_path),
            "--target-pdb", str(self.target.pdb_path),
            "--vllm-base-url", self.config.llm.base_url,
            "--llm-model", self.config.llm.model,
            "--max-wall-h", str(run.max_wall_hours),
            "--worker-gpus", ",".join(run.worker_gpus),
            "--enabled-families", ",".join(self.effective_families),
            "--enable-critic", str(int(policy.critic)),
            "--enable-evidence-skip", str(int(policy.evidence_skip)),
            "--enable-exemplars", str(int(policy.exemplars)),
            "--foldseek-su-tm-score", str(policy.foldseek_su_tm_score),
            "--foldseek-collapse-tm-score", str(policy.foldseek_collapse_tm_score),
            "--seed", str(run.seed),
            "--selector-quota-realization", policy.selector_quota_realization,
            "--selector-mode-window-k", str(policy.selector_mode_window_k),
            "--selector-adaptive-mode-window-k",
            str(int(policy.selector_adaptive_mode_window_k)),
        ]

    def environment(self) -> dict[str, str]:
        """Return the complete compatibility environment for one controller run."""

        env = self.config.backends.environment()
        if self.config.memory.cross_campaign_path is not None:
            env["TREX_CROSS_CAMPAIGN_MEMORY"] = str(
                self.config.memory.cross_campaign_path
            )
        env.update({
            "TREX_WORKER_GPUS": ",".join(self.config.run.worker_gpus),
            "TREX_ENABLED_FAMILIES": ",".join(self.effective_families),
        })
        return env

    def runtime_paths(self) -> RuntimePaths:
        """Return the typed backend-path input passed to the controller."""

        return self.config.backends.to_runtime_paths()

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical, human-inspectable resolved input artifact."""

        run = self.config.run
        policy = self.config.policy
        configured_environment = self.environment()
        return {
            "schema_version": RESOLVED_SCHEMA_VERSION,
            "campaign": {
                "name": self.config.name,
                "source_path": str(self.source_path),
                "source_sha256": self.source_sha256,
                "resolution_notes": list(self.resolution_notes),
            },
            "target": self.target.as_json(),
            "run": {
                "archive_root": str(run.archive_root),
                "max_wall_hours": run.max_wall_hours,
                "seed": run.seed,
                "worker_gpus": list(run.worker_gpus),
                "enabled_families": list(self.effective_families),
            },
            "llm": {
                "base_url": self.config.llm.base_url,
                "model": self.config.llm.model,
            },
            "policy": {
                "critic": policy.critic,
                "evidence_skip": policy.evidence_skip,
                "exemplars": policy.exemplars,
                "foldseek_su_tm_score": policy.foldseek_su_tm_score,
                "foldseek_collapse_tm_score": policy.foldseek_collapse_tm_score,
                "selector_quota_realization": policy.selector_quota_realization,
                "selector_mode_window_k": policy.selector_mode_window_k,
                "selector_adaptive_mode_window_k": policy.selector_adaptive_mode_window_k,
            },
            "memory": {
                "cross_campaign_path": (
                    str(self.config.memory.cross_campaign_path)
                    if self.config.memory.cross_campaign_path is not None
                    else None
                ),
            },
            "backends": self.config.backends.as_dict(),
            "controller_interface": {
                "argv": self.controller_argv(),
                "runtime_paths": self.runtime_paths().as_dict(),
                "configured_environment": configured_environment,
                "inherited_trex_environment": inspectable_environment(
                    os.environ,
                    prefix="TREX_",
                    excluded_names=frozenset(configured_environment),
                ),
            },
        }
