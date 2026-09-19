"""Configuration contracts shared by the live-tick phases.

The public names continue to be re-exported from :mod:`trex.live_tick` for
backward compatibility.  Keeping the contracts here lets individual phases
depend on configuration without importing the orchestration module.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..candidate_builder import BuilderConfig
from ..critic import CriticCallConfig
from ..lifecycle import LifecycleConfig
from ..planner import PlannerCallConfig
from ..selector import SelectorConfig
from ..supervisor import SupervisorCallConfig


@dataclass(frozen=True)
class FoldseekConfig:
    """Foldseek structural-clustering configuration.

    Official live SU uses strict-only Foldseek clusters at ``min_tm_score``.
    Whole-archive collapse and near-miss signals use ``collapse_tm_score``.
    """

    enabled: bool = True
    binary: str = "foldseek"
    min_tm_score: float = 0.60
    collapse_tm_score: float = 0.60
    timeout_seconds: int = 600
    whole_archive_window: int = 400
    strict_tm08_diagnostic_enabled: bool = False
    fine_tm_score: float = 0.80
    fine_strict_window: int = 160
    fine_min_strict: int = 4
    fine_refresh_every_ticks: int = 5
    near_miss_refresh_every_ticks: int = 5


@dataclass(frozen=True)
class SequenceDedupConfig:
    """MMseqs2 configuration for secondary sequence-diversity evidence."""

    enabled: bool = False
    binary: str = "mmseqs"
    min_seq_id: float = 0.90
    coverage: float = 0.80
    timeout_seconds: int = 600
    binder_chain_id: str = "B"


@dataclass(frozen=True)
class SkipConfig:
    """Reuse the previous decision when decision-relevant evidence is unchanged."""

    enabled: bool = False


@dataclass(frozen=True)
class LiveTickConfig:
    """Top-level configuration for one scientific live tick."""

    available_slots: int = 3
    worker_wall_gpu_count: int = 3
    window_size: int = 60
    planner: PlannerCallConfig = PlannerCallConfig()
    supervisor: SupervisorCallConfig = SupervisorCallConfig()
    unified_reasoner: bool = False
    selector: SelectorConfig = SelectorConfig()
    builder: BuilderConfig = BuilderConfig()
    lifecycle: LifecycleConfig = LifecycleConfig()
    critic: CriticCallConfig = CriticCallConfig()
    foldseek: FoldseekConfig = FoldseekConfig()
    sequence_dedup: SequenceDedupConfig = SequenceDedupConfig()
    skip: SkipConfig = SkipConfig()
    enable_exemplars: bool = True


__all__ = [
    "FoldseekConfig",
    "LiveTickConfig",
    "SequenceDedupConfig",
    "SkipConfig",
]
