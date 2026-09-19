"""Worker-slot state with no backend or archive responsibilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from ...schemas import ActionCandidate


class PollableProcess(Protocol):
    """Minimal subprocess surface needed by a worker slot."""

    def poll(self) -> int | None: ...


@dataclass
class WorkerSlot:
    """One worker GPU and its current asynchronous job state.

    The lifecycle is ``idle -> busy -> done -> free``. Parsing, archive writes,
    timeout decisions, and dispatch policy intentionally live elsewhere.
    """

    slot_id: int
    gpu_id: str
    proc: PollableProcess | None = None
    cand: ActionCandidate | None = None
    out_dir: Path | None = None
    parent_pdb_str: str = ""
    parent_result_id: str = ""
    target_chains_csv: str = ""
    binder_chain: str = ""
    tick_id: str = ""
    launched_at: float = 0.0
    last_yield_at: float = 0.0
    prev_yield_n: int = -1
    last_incremental_parse_at: float = 0.0
    archived_elapsed_gpu_h: float = 0.0
    archived_result_ids: set[str] = field(default_factory=set)

    @property
    def busy(self) -> bool:
        return self.proc is not None

    def is_done(self) -> bool:
        return self.proc is not None and self.proc.poll() is not None

    def free(self) -> None:
        """Return the slot to a clean idle state after completion or salvage."""

        self.proc = None
        self.cand = None
        self.out_dir = None
        self.parent_pdb_str = ""
        self.parent_result_id = ""
        self.target_chains_csv = ""
        self.binder_chain = ""
        self.tick_id = ""
        self.launched_at = 0.0
        self.last_yield_at = 0.0
        self.prev_yield_n = -1
        self.last_incremental_parse_at = 0.0
        self.archived_elapsed_gpu_h = 0.0
        self.archived_result_ids.clear()
