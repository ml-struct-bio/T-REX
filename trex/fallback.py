"""State-conditioned fallback mixtures and allocation bounds.

Used for invalid, unavailable, or empty LLM decisions and fallback-only comparisons. Low
confidence can activate bounds without invalidating an otherwise valid output.
"""

from __future__ import annotations

from dataclasses import dataclass

from .schemas import StateLabel


CONFIDENT_MIXTURE_THRESHOLD = 0.65


DEFAULT_MIXTURES: dict[StateLabel, dict[str, float]] = {
    "productive": {"exploit": 0.65, "rescue": 0.25, "explore": 0.10},
    # Retain exploitation while allowing refinement of a productive but repetitive
    # route.
    "productive_duplicate": {"exploit": 0.55, "rescue": 0.35, "explore": 0.10},
    # strict_duplicate_collapse: raw strict success is cheap to reproduce but not
    # turning into official SU. Stop polishing the same structural basin; spend
    # most budget on fresh roots/families/configs, with a small rescue lane only
    # for material, blocker-changing repairs.
    "strict_duplicate_collapse": {"exploit": 0.10, "rescue": 0.15, "explore": 0.75},
    "rescue_rich": {"exploit": 0.30, "rescue": 0.50, "explore": 0.20},
    "stalled": {"exploit": 0.20, "rescue": 0.35, "explore": 0.45},
    # Bias prolonged stalls toward exploration while dispatch throttles control
    # evaluation backlogs.
    "deep_stall": {"exploit": 0.20, "rescue": 0.30, "explore": 0.50},
    "low_evidence": {"exploit": 0.35, "rescue": 0.30, "explore": 0.35},
}


@dataclass(frozen=True)
class Clamp:
    """Per-state clamp on mode mixture. None means no constraint."""

    exploit_min: float | None = None
    exploit_max: float | None = None
    rescue_min: float | None = None
    rescue_max: float | None = None
    explore_min: float | None = None
    explore_max: float | None = None


DEFAULT_CLAMPS: dict[StateLabel, Clamp] = {
    # Limit exploration in productive campaigns while retaining a nonzero exploration share.
    "productive": Clamp(
        exploit_min=0.45,
        exploit_max=0.80,
        rescue_min=0.10,
        explore_min=0.05,
        explore_max=0.20,
    ),
    "productive_duplicate": Clamp(
        exploit_min=0.40,
        exploit_max=0.70,
        rescue_min=0.20,
        explore_min=0.05,
        explore_max=0.30,
    ),
    "strict_duplicate_collapse": Clamp(
        exploit_max=0.20,
        rescue_max=0.30,
        explore_min=0.55,
    ),
    "deep_stall": Clamp(exploit_max=0.30, rescue_min=0.15, explore_min=0.45),
    # Rescue-rich: rescue is the primary path; cap explore so the LLM
    # doesn't pivot away from the rescue lane that the evidence supports.
    "rescue_rich": Clamp(rescue_min=0.40, explore_max=0.30),
    # Stalled: explore is desired here. No upper bound on explore.
    "stalled": Clamp(exploit_max=0.35, rescue_min=0.25, explore_min=0.25),
    "low_evidence": Clamp(explore_min=0.05),
}


def describe_clamp_for_prompt(state: "StateLabel") -> str:
    """Render Supervisor allocation guidance from the same bounds used by the controller."""
    c = DEFAULT_CLAMPS.get(state, Clamp())
    parts: list[str] = []
    for mode, lo, hi in (
        ("exploit", c.exploit_min, c.exploit_max),
        ("rescue", c.rescue_min, c.rescue_max),
        ("explore", c.explore_min, c.explore_max),
    ):
        if lo is None and hi is None:
            continue
        if lo is not None and hi is not None:
            parts.append(f"{mode} {lo:.2f}-{hi:.2f}")
        elif lo is not None:
            parts.append(f"{mode} >= {lo:.2f}")
        else:
            parts.append(f"{mode} <= {hi:.2f}")
    if not parts:
        return "no hard range for this state — you decide, but keep a small explore floor (>=0.05) for diversity insurance"
    return "; ".join(parts)


# Unconditional bounds leave allocation freedom within minimum safeguards. Stronger
# state-dependent bounds apply only when their triggers fire.
CATEGORY_A_CLAMPS: dict[StateLabel, Clamp] = {
    # Productive: keep a tiny exploit ceiling so a degenerate
    # mixture (e.g. {exploit:1.0}) never goes through, and a tiny
    # explore floor so the diversity insurance is preserved.
    "productive": Clamp(exploit_max=0.95, explore_min=0.05),
    "productive_duplicate": Clamp(exploit_max=0.90, explore_min=0.05),
    "strict_duplicate_collapse": Clamp(exploit_max=0.30, explore_min=0.45),
    "rescue_rich": Clamp(rescue_min=0.10),
    # Stalled keeps the full DEFAULT_CLAMPS unconditionally — its
    # clamps ARE the diversity-collapse response. See
    # `should_apply_category_b_clamps` (returns True for stalled).
    "stalled": Clamp(exploit_max=0.35, rescue_min=0.25, explore_min=0.25),
    # Maintain an exploration floor during deep stall.
    "deep_stall": Clamp(exploit_max=0.40, explore_min=0.35),
    "low_evidence": Clamp(explore_min=0.05),
}

# Raise the exploration floor when structural concentration crosses this threshold.
DIVERSITY_COLLAPSE_TOP_BIN = 0.50
DIVERSITY_COLLAPSE_EXPLORE_FLOOR = 0.20
# rescue_rich uses a deliberately HIGHER collapse bar than productive states: it
# already runs explore-light by design, so only loosen explore on a strong
# collapse. Named (was a bare 0.70) to make the independence from the productive
# threshold explicit.
RESCUE_RICH_COLLAPSE_TOP_BIN = 0.70


def should_apply_category_b_clamps(
    state: StateLabel,
    *,
    top_bin_share: float | None,
    strict_su_top_bin_share: float | None = None,
    panel_ready_bins_covered: int = 0,
    panel_size_K: int = 8,
    supervisor_confidence: float | None,
    recent_fallback_high: bool,
    supervisor_used: bool,
    panel_live: bool = False,
) -> tuple[bool, list[str]]:
    """Return whether conditional allocation bounds apply and the triggering reasons.

    Fallback and stalled states always apply them. Otherwise evaluate structural
    concentration, enabled live-panel coverage, Supervisor confidence, and recent
    fallback frequency. Without a trigger, use the unconditional safety bounds.
    """
    reasons: list[str] = []
    # Fallback path: no LLM mixture to respect, always clamp.
    if not supervisor_used:
        return True, ["fallback_path"]
    # Always apply the stalled-state clamps. For deep_stall, apply conditional clamps
    # only when the evidence or confidence checks below require them.
    if state == "stalled":
        return True, ["state_stalled"]
    if state == "strict_duplicate_collapse":
        return True, ["state_strict_duplicate_collapse"]
    # Collapse / low-confidence triggers. Use the stronger of all-scored
    # collapse and strict-SU collapse: the former catches failure-mode
    # convergence, the latter catches mined-out winners.
    collapse_share = top_bin_share
    collapse_label = "top_bin_share"
    if strict_su_top_bin_share is not None and (
        collapse_share is None or strict_su_top_bin_share > collapse_share
    ):
        collapse_share = strict_su_top_bin_share
        collapse_label = "strict_su_top_bin_share"
    if collapse_share is not None and collapse_share >= DIVERSITY_COLLAPSE_TOP_BIN:
        reasons.append(
            f"{collapse_label}={collapse_share:.2f}>={DIVERSITY_COLLAPSE_TOP_BIN}"
        )
    if panel_live and panel_ready_bins_covered < (panel_size_K // 2):
        reasons.append(f"panel_bins={panel_ready_bins_covered}<K/2={panel_size_K//2}")
    if (
        supervisor_confidence is not None
        and supervisor_confidence < CONFIDENT_MIXTURE_THRESHOLD
    ):
        reasons.append(
            f"sup_conf={supervisor_confidence:.2f}<{CONFIDENT_MIXTURE_THRESHOLD:.2f}"
        )
    if recent_fallback_high:
        reasons.append("recent_fb>=0.30")
    return (bool(reasons), reasons)


def diversity_adjusted_clamp(
    base: Clamp,
    state: StateLabel,
    *,
    top_bin_share: float | None,
    strict_su_top_bin_share: float | None = None,
    panel_ready_bins_covered: int = 0,
    panel_size_K: int = 8,
    panel_live: bool = False,
) -> tuple[Clamp, list[str]]:
    """Loosen clamps when diversity signals indicate collapse.

    Answer to user Q ("can clamps respect LLM agency when diversity matters?"):
    pure state→clamp lookup is too rigid. When the productive run is
    structurally collapsing (top Foldseek bin holds >= DIVERSITY_COLLAPSE_TOP_BIN
    = 50% of recent successes), the original exploit_max=0.80 / explore_max=0.20
    over-favors more-of-the-same. Force an explore floor + drop exploit_max so the
    controller pivots toward diversity (not just permits it).

    Returns (adjusted_clamp, applied_adjustments_log). Adjustments are
    additive over the base clamp — base safety floors remain (we never
    move exploit_min below productive's 0.45 floor, etc.).
    """
    adjustments: list[str] = []
    # Productive structural-collapse signal: top-bin share dominant AND
    # panel bin coverage is shallow relative to target panel size. Prefer the
    # strict-SU collapse signal when it is stronger, but keep the all-scored
    # signal as the fallback for no-strict or general convergence regimes.
    collapse_share = top_bin_share
    collapse_label = "top_bin"
    if strict_su_top_bin_share is not None and (
        collapse_share is None or strict_su_top_bin_share > collapse_share
    ):
        collapse_share = strict_su_top_bin_share
        collapse_label = "strict_su_top"
    if state in ("productive", "productive_duplicate") and collapse_share is not None:
        # Use panel coverage only when live panel selection is enabled.
        bin_coverage_short = panel_live and panel_ready_bins_covered < panel_size_K // 2
        if collapse_share >= DIVERSITY_COLLAPSE_TOP_BIN or bin_coverage_short:
            # Raise the exploration floor under structural concentration. An exploitation cap
            # of 0.60 leaves room for exploration and rescue floors of 0.20 each.
            adjusted = Clamp(
                exploit_min=base.exploit_min,
                exploit_max=0.60 if base.exploit_max else None,
                rescue_min=base.rescue_min,
                rescue_max=base.rescue_max,
                explore_min=max(
                    base.explore_min or 0.0, DIVERSITY_COLLAPSE_EXPLORE_FLOOR
                ),
                explore_max=0.40 if base.explore_max else None,
            )
            adjustments.append(
                f"diversity_loosen({collapse_label}={collapse_share:.2f}>={DIVERSITY_COLLAPSE_TOP_BIN},"
                f"explore_floor={DIVERSITY_COLLAPSE_EXPLORE_FLOOR},"
                f"panel_bins={panel_ready_bins_covered})"
            )
            return adjusted, adjustments
    # Rescue-rich: if panel diversity is also collapsing, allow more explore
    if (
        state == "rescue_rich"
        and collapse_share is not None
        and collapse_share >= RESCUE_RICH_COLLAPSE_TOP_BIN
    ):
        adjusted = Clamp(
            rescue_min=base.rescue_min,
            explore_max=0.50,
        )
        adjustments.append(
            f"diversity_loosen_rescue({collapse_label}={collapse_share:.2f})"
        )
        return adjusted, adjustments
    return base, adjustments


def normalize(mixture: dict[str, float]) -> dict[str, float]:
    s = sum(max(0.0, v) for v in mixture.values())
    if s <= 0:
        return {"exploit": 1 / 3, "rescue": 1 / 3, "explore": 1 / 3}
    return {k: max(0.0, v) / s for k, v in mixture.items()}


def clamp_mixture(
    mixture: dict[str, float],
    state: StateLabel,
    *,
    all_explore_backends_unhealthy: bool = False,
    route_backlog_saturated: bool = False,
    top_bin_share: float | None = None,
    strict_su_top_bin_share: float | None = None,
    panel_ready_bins_covered: int = 0,
    panel_size_K: int = 8,
    category_b_enabled: bool = True,
    panel_live: bool = False,
) -> tuple[dict[str, float], list[str]]:
    """Apply allocation bounds while preserving normalization.

    Pin out-of-range shares and redistribute remaining mass proportionally until stable.
    Conditional bounds may loosen under structural concentration. Return the mixture and
    applied-bound log.
    """
    if category_b_enabled:
        base_clamp = DEFAULT_CLAMPS.get(state, Clamp())
        clamp, diversity_log = diversity_adjusted_clamp(
            base_clamp,
            state,
            top_bin_share=top_bin_share,
            strict_su_top_bin_share=strict_su_top_bin_share,
            panel_ready_bins_covered=panel_ready_bins_covered,
            panel_size_K=panel_size_K,
            panel_live=panel_live,
        )
        log: list[str] = list(diversity_log)
    else:
        # Unconditional bounds do not need conditional diversity adjustments.
        clamp = CATEGORY_A_CLAMPS.get(state, Clamp())
        log = ["category_b_gated_off"]

    bounds: dict[str, tuple[float | None, float | None]] = {
        "exploit": (clamp.exploit_min, clamp.exploit_max),
        "rescue": (clamp.rescue_min, clamp.rescue_max),
        "explore": (clamp.explore_min, clamp.explore_max),
    }
    # waivers
    if all_explore_backends_unhealthy:
        lo, hi = bounds["explore"]
        bounds["explore"] = (None, hi)
    if route_backlog_saturated:
        lo, hi = bounds["rescue"]
        bounds["rescue"] = (None, hi)

    # Normalize before comparing shares with fractional bounds.
    _clipped = {k: max(0.0, v) for k, v in mixture.items()}
    _tot = sum(_clipped.values())
    out = {k: v / _tot for k, v in _clipped.items()} if _tot > 0 else _clipped
    # Include every mode so incomplete input mixtures remain well-defined.
    out = {m: out.get(m, 0.0) for m in bounds}
    pinned: dict[str, float] = {}
    for mode, (lo, hi) in bounds.items():
        v = out.get(mode, 0.0)
        if lo is not None and v < lo:
            pinned[mode] = lo
            log.append(f"clamp:{mode}_min({lo})")
        elif hi is not None and v > hi:
            pinned[mode] = hi
            log.append(f"clamp:{mode}_max({hi})")

    for _iter in range(5):
        free_modes = [m for m in bounds if m not in pinned]
        pinned_sum = sum(pinned.values())
        if pinned_sum > 1.0:
            # Infeasible floors; renormalize pinned alone.
            s = pinned_sum
            return (
                {m: v / s for m, v in pinned.items()} | {m: 0.0 for m in free_modes}
            ), log

        # If all pinned shares sum below one, scale them and recheck upper bounds.
        if not free_modes and pinned_sum > 0 and pinned_sum < 1.0:
            scale = 1.0 / pinned_sum
            candidate = {m: v * scale for m, v in pinned.items()}
            # Re-clamp against max bounds (scaling could push above max);
            # if any mode exceeds its max, pin to max and try to redistribute
            # the residual across modes still below their max.
            need_more = []
            overshoot = 0.0
            for mode, val in candidate.items():
                _, hi = bounds[mode]
                if hi is not None and val > hi:
                    overshoot += val - hi
                    candidate[mode] = hi
                else:
                    need_more.append(mode)
            if overshoot > 0 and need_more:
                # Distribute overshoot to modes still below their cap.
                share = overshoot / len(need_more)
                for m in need_more:
                    _, hi = bounds[m]
                    new_val = candidate[m] + share
                    if hi is not None and new_val > hi:
                        candidate[m] = hi
                    else:
                        candidate[m] = new_val
            log.append(f"all_pinned_scale(pinned_sum={pinned_sum:.2f})")
            return candidate, log

        free_total_input = sum(out[m] for m in free_modes)
        free_target_mass = 1.0 - pinned_sum
        if free_total_input <= 0 and free_modes:
            # equal split for the remaining mass
            split = free_target_mass / len(free_modes)
            new_free = {m: split for m in free_modes}
        else:
            new_free = {
                m: out[m] * (free_target_mass / free_total_input) for m in free_modes
            }

        candidate = dict(pinned)
        candidate.update(new_free)

        # Check whether scaling pushed any free mode out of its bound.
        newly_pinned = False
        for mode in free_modes:
            lo, hi = bounds[mode]
            if lo is not None and candidate[mode] < lo:
                pinned[mode] = lo
                log.append(f"clamp:{mode}_min({lo})")
                newly_pinned = True
            elif hi is not None and candidate[mode] > hi:
                pinned[mode] = hi
                log.append(f"clamp:{mode}_max({hi})")
                newly_pinned = True
        if not newly_pinned:
            return candidate, log
        # else: rerun loop with the expanded pin set
        for mode in free_modes:
            if mode not in pinned:
                out[mode] = candidate[mode]

    # Fixed-point not reached in 5 iters; return current candidate normalized
    return normalize(candidate), log


def fallback_mixture(
    state: StateLabel,
    *,
    all_explore_backends_unhealthy: bool = False,
    route_backlog_saturated: bool = False,
) -> tuple[dict[str, float], list[str]]:
    base = DEFAULT_MIXTURES[state]
    return clamp_mixture(
        base,
        state,
        all_explore_backends_unhealthy=all_explore_backends_unhealthy,
        route_backlog_saturated=route_backlog_saturated,
    )


def largest_remainder(mixture: dict[str, float], total_slots: int) -> dict[str, int]:
    """Largest-remainder rounding for slot quotas.

    Guarantees sum(quotas) == total_slots when total_slots >= 0.
    """
    if total_slots <= 0:
        return {k: 0 for k in mixture}
    raw = {k: v * total_slots for k, v in mixture.items()}
    floor = {k: int(v) for k, v in raw.items()}
    remainder = total_slots - sum(floor.values())
    # distribute remainder by largest fractional part, then by stable key order
    order = sorted(raw.items(), key=lambda kv: (-(kv[1] - int(kv[1])), kv[0]))
    for k, _ in order[:remainder]:
        floor[k] += 1
    return floor


def redistribute_empty_modes(
    quotas: dict[str, int],
    feasibility_by_mode: dict[str, bool],
    mixture: dict[str, float],
) -> tuple[dict[str, int], list[str]]:
    """Move quota from infeasible modes to feasible ones proportionally.

    feasibility_by_mode[m] = True iff at least one feasible candidate exists for m.
    """
    log: list[str] = []
    feasible_modes = [m for m, ok in feasibility_by_mode.items() if ok]
    if not feasible_modes:
        return quotas, log

    moved_total = 0
    out = dict(quotas)
    for m in list(out.keys()):
        if m in feasible_modes:
            continue
        if out[m] > 0:
            log.append(f"redistribute:{m}->{feasible_modes}({out[m]})")
            moved_total += out[m]
            out[m] = 0

    if moved_total == 0:
        return out, log

    # proportionally to mixture weight among feasible modes
    feasible_weights = {m: max(0.0, mixture.get(m, 0.0)) for m in feasible_modes}
    w_sum = sum(feasible_weights.values())
    if w_sum <= 0:
        # equal split
        feasible_weights = {m: 1.0 for m in feasible_modes}
        w_sum = float(len(feasible_modes))

    add = largest_remainder(
        {m: w / w_sum for m, w in feasible_weights.items()}, moved_total
    )
    for m, n in add.items():
        out[m] += n

    # Preserve representation of feasible modes with a nontrivial proposed share.
    # If rounding gives one such mode no slots, transfer one from the feasible mode
    # with the largest quota, provided that donor has more than one slot.
    MIN_REPRESENTATION_FLOOR = 0.08
    protected = [
        m
        for m in feasible_modes
        if mixture.get(m, 0.0) >= MIN_REPRESENTATION_FLOOR and out[m] == 0
    ]
    for m in protected:
        donor = max(
            (k for k in feasible_modes if k != m and out[k] > 1),
            key=lambda k: out[k],
            default=None,
        )
        if donor is None:
            break  # nothing to steal from
        out[donor] -= 1
        out[m] += 1
        log.append(f"min_representation:{m}<-{donor}(mixture={mixture[m]:.2f})")
    return out, log
