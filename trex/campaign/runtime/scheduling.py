"""Small scheduling decisions independent of scientific policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol


@dataclass(frozen=True)
class ControllerLoopConfig:
    """Operational loop settings; ``maximum_iterations`` is for bounded runs."""

    poll_interval_seconds: float = 5.0
    state_probe_cache_seconds: float = 60.0
    progress_checkpoint_interval_seconds: float = 60.0
    maximum_iterations: int | None = None

    def __post_init__(self) -> None:
        if self.poll_interval_seconds < 0:
            raise ValueError("poll_interval_seconds must be non-negative")
        if self.state_probe_cache_seconds < 0:
            raise ValueError("state_probe_cache_seconds must be non-negative")
        if self.progress_checkpoint_interval_seconds < 0:
            raise ValueError(
                "progress_checkpoint_interval_seconds must be non-negative"
            )
        if self.maximum_iterations is not None and self.maximum_iterations < 1:
            raise ValueError("maximum_iterations must be at least 1 when configured")


class BusySlot(Protocol):
    @property
    def busy(self) -> bool: ...


def controller_sleep_seconds(
    *,
    dispatched: int,
    planned: int,
    pending: list[str],
    pool: Iterable[BusySlot],
    poll_interval_s: float,
) -> float:
    """Use long backoff only when no queued or running work exists."""

    any_busy = any(slot.busy for slot in pool)
    if dispatched == 0 and planned == 0 and not pending and not any_busy:
        return 120.0
    return max(0.0, float(poll_interval_s))
