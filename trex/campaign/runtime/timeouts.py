"""Worker timeout policy, separate from process management and parsing."""

from __future__ import annotations

import os


# Historical expected-runtime references retained for compatibility and audit.
FAMILY_TIMEOUT_S: dict[str, int] = {
    "bindcraft": 9000,
    "complexa_beam": 1800,
    "complexa_best_of_n": 1800,
    "complexa_fk_steering": 1800,
    "complexa_mcts": 3600,
    "structure_refilter": 900,
    "proteinmpnn_redesign": 300,
    "boltzgen": 7200,
}

# Generous hang backstops. BindCraft uses the progress watchdog below rather
# than its reference ceiling because a productive long run should keep going.
HARD_CEILING_S: dict[str, int] = {
    "bindcraft": 18000,
    "complexa_mcts": 9000,
    "complexa_beam": 5400,
    "complexa_best_of_n": 5400,
    "complexa_fk_steering": 5400,
    "structure_refilter": 2160,
    "proteinmpnn_redesign": 900,
    "boltzgen": 10800,
}
DEFAULT_HARD_CEILING_S = 7200
BINDCRAFT_FLOOR_S = 5400
BINDCRAFT_YIELD_WINDOW_S = 5400

try:
    BINDCRAFT_INCREMENTAL_PARSE_INTERVAL_S = max(
        300.0,
        float(os.environ.get("TREX_BINDCRAFT_INCREMENTAL_PARSE_INTERVAL_S", "600")),
    )
except ValueError:
    BINDCRAFT_INCREMENTAL_PARSE_INTERVAL_S = 600.0


def worker_timeout_reason(
    family: str,
    *,
    elapsed_slot_s: float,
    ceiling_s: float,
    now_t: float,
    accepted_count: int | None = None,
    progress_count: int | None = None,
    prev_yield_n: int = -1,
    last_yield_at: float = 0.0,
) -> tuple[str | None, int, float]:
    """Return a timeout reason and updated BindCraft progress state."""

    if family == "bindcraft":
        if elapsed_slot_s <= BINDCRAFT_FLOOR_S:
            return None, prev_yield_n, last_yield_at
        progress_n = progress_count if progress_count is not None else accepted_count
        progress_n = progress_n if progress_n is not None else 0
        if prev_yield_n < 0:
            return None, progress_n, now_t
        if progress_n > prev_yield_n:
            return None, progress_n, now_t
        if (now_t - last_yield_at) > BINDCRAFT_YIELD_WINDOW_S:
            return "bindcraft-no-yield", prev_yield_n, last_yield_at
        return None, prev_yield_n, last_yield_at

    if elapsed_slot_s > ceiling_s:
        return "hard-ceiling", prev_yield_n, last_yield_at
    return None, prev_yield_n, last_yield_at
