"""Deterministic fallback: state-conditioned mode mixtures + clamps.

Used when:
  - planner/supervisor LLM output is invalid, abstaining, empty, or timed out
  - LLM backend is in cooldown
  - testing fallback-only arm (Phase 7)

Low confidence does not invalidate an otherwise valid output; it is retained and
may activate conditional deterministic bounds.

The state classifier itself is in `evidence_reducer.py` (closer to the
reduction logic). This module owns: (a) state → default mixture lookup,
(b) clamps, (c) within-fallback ranking helpers.
"""

from __future__ import annotations

from dataclasses import dataclass

from .schemas import StateLabel


CONFIDENT_MIXTURE_THRESHOLD = 0.65


DEFAULT_MIXTURES: dict[StateLabel, dict[str, float]] = {
    "productive": {"exploit": 0.65, "rescue": 0.25, "explore": 0.10},
    # productive_duplicate (F6): winning generator but duplicate-collapsing —
    # keep exploiting it, push the freed mass to rescue (intra-family structural
    # diversification: sc_scale_noise / beam_width / epitope-contact tweaks), NOT
    # cross-family explore. Strictly exploit-leaning so a fallback tick over a
    # producing lane does not bleed budget off the family that is buying SU.
    "productive_duplicate": {"exploit": 0.55, "rescue": 0.35, "explore": 0.10},
    # strict_duplicate_collapse: raw strict success is cheap to reproduce but not
    # turning into official SU. Stop polishing the same structural basin; spend
    # most budget on fresh roots/families/configs, with a small rescue lane only
    # for material, blocker-changing repairs.
    "strict_duplicate_collapse": {"exploit": 0.10, "rescue": 0.15, "explore": 0.75},
    "rescue_rich": {"exploit": 0.30, "rescue": 0.50, "explore": 0.20},
    "stalled": {"exploit": 0.20, "rescue": 0.35, "explore": 0.45},
    # deep_stall (F5): long no-new-SU plateau. SOFT explore lean, NOT a hard
    # override — the archive analysis showed the real escape blocker on collapsed
    # targets was FEASIBILITY (route_cap saturated by the refilter flood blocked
    # explore candidates: CbAgo explore target 0.37 realised only 0.24, 100/372
    # explore candidates rejected on route_cap_ok), not the explore RATIO. The
    # deterministic chain throttle drains that backlog so the LLM's already-
    # explore-leaning mixture becomes realisable; this state only adds a floor.
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
    # Productive: V6.3 lesson — CD45-style targets were hurt by forced
    # exploration. Cap explore at 0.20 so LLM creep cannot waste budget on
    # novel methods when the current family is producing structure-unique
    # successes. Small explore floor (0.05) preserves diversity insurance.
    "productive": Clamp(
        exploit_min=0.45,
        exploit_max=0.80,
        rescue_min=0.10,
        explore_min=0.05,
        explore_max=0.20,
    ),
    # productive_duplicate (F6): keep exploit hot (the lane is buying SU) but give
    # rescue room for intra-family structural diversification; cap explore so the
    # LLM doesn't pivot to novel families when the issue is structural redundancy.
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
    # deep_stall (F5) Category-B (fires on confirmed collapse/low-conf): firmer
    # explore floor once collapse is corroborated, but still not a hard override.
    "deep_stall": Clamp(exploit_max=0.30, rescue_min=0.15, explore_min=0.45),
    # Rescue-rich: rescue is the primary path; cap explore so the LLM
    # doesn't pivot away from the rescue lane that the evidence supports.
    "rescue_rich": Clamp(rescue_min=0.40, explore_max=0.30),
    # Stalled: explore is desired here. No upper bound on explore.
    "stalled": Clamp(exploit_max=0.35, rescue_min=0.25, explore_min=0.25),
    # Low evidence: let the LLM decide, but keep ONLY a tiny explore floor (v7_3)
    # so the diversity-insurance budget is realizable on BOTH the Category-A and
    # Category-B paths. Previously empty Clamp() — the only state where a
    # sub-~0.05 explore share passed unclamped and then K-window-starved to 0.
    "low_evidence": Clamp(explore_min=0.05),
}


def describe_clamp_for_prompt(state: "StateLabel") -> str:
    """Human-readable mode_mixture range guidance for the Supervisor prompt.

    Option C (2026-05-28): the Supervisor is now asked to emit a mixture that
    ALREADY respects the per-state strategy ranges, so the deterministic clamp
    becomes a rarely-binding safety net rather than a frequent override. Single
    source of truth = DEFAULT_CLAMPS, so the prompt guidance and the enforced
    clamp can never drift apart.
    """
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


# §22.8.4: Category A clamps — bug+budget safety floors only.
# Applied unconditionally on every supervisor output. These guarantee
# the launched mixture is never pathological (no all-exploit-no-explore
# in productive, no all-explore-no-exploit in productive, etc.) but
# leave room for the LLM to express high-conviction allocations
# (e.g., {exploit:0.90, rescue:0.05, explore:0.05}).
#
# The wider DEFAULT_CLAMPS above are Category B (strategy-shaping) and
# are now only applied when the gating signals fire (see
# `should_apply_category_b_clamps`).
CATEGORY_A_CLAMPS: dict[StateLabel, Clamp] = {
    # Productive: keep a tiny exploit ceiling so a degenerate
    # mixture (e.g. {exploit:1.0}) never goes through, and a tiny
    # explore floor so the diversity insurance is preserved.
    "productive": Clamp(exploit_max=0.95, explore_min=0.05),
    # productive_duplicate (F6): same degenerate-prevention floors as productive —
    # the LLM owns the diversify-in-place split within these loose bounds.
    "productive_duplicate": Clamp(exploit_max=0.90, explore_min=0.05),
    "strict_duplicate_collapse": Clamp(exploit_max=0.30, explore_min=0.45),
    "rescue_rich": Clamp(rescue_min=0.10),
    # Stalled keeps the full DEFAULT_CLAMPS unconditionally — its
    # clamps ARE the diversity-collapse response. See
    # `should_apply_category_b_clamps` (returns True for stalled).
    "stalled": Clamp(exploit_max=0.35, rescue_min=0.25, explore_min=0.25),
    # deep_stall (F5) Category-A always-on insurance: a 0.35 explore FLOOR so the
    # slots freed by the chain throttle reliably go to fresh exploration rather
    # than re-exploiting the dead lane — but the confident LLM's mixture is still
    # honored above this floor (deep_stall is NOT in the always-Cat-B set), since
    # the analysis shows the LLM reacts correctly under collapse (it already
    # targets high explore); the throttle + this floor are the insurance.
    "deep_stall": Clamp(exploit_max=0.40, explore_min=0.35),
    # low_evidence (v7_3 2026-06-10): a tiny explore floor (matching productive)
    # so the diversity-insurance budget is REALIZABLE. Previously empty Clamp(),
    # which was the ONLY state where a sub-~0.05 explore share passed through
    # unclamped and then K-window-starved to exactly 0 (the residual quantization
    # bug). 0.05 reliably realizes ~1 explore launch per ~10-20 ticks. In a
    # cold/uncertain state a guaranteed minimum exploration is the desired prior.
    "low_evidence": Clamp(explore_min=0.05),
}

# Diversity-collapse response (2026-06-13). The old collapse trigger was 0.70,
# but live CD45 (9630421) collapsed to top_bin_share ~0.44-0.50 while explore was
# held at 0.10-0.15 (productive_duplicate floor 0.05) — the controller SAW the
# collapse but under-responded, re-mining one Foldseek basin. Trigger the
# diversity response at 0.50 and FORCE an explore floor (raise explore_min, not
# just loosen the cap) so budget actually moves off the collapsing basin.
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
    """§22.8.4 gating policy.

    Returns (apply, reasons). When `apply` is False, the Selector uses
    `CATEGORY_A_CLAMPS` (bug+budget only) — honoring the LLM's mixture
    within ±safety floors. When True, the full `DEFAULT_CLAMPS` set
    (strategy-shaping) applies as before.

    Always True for:
      - `stalled` state (the Category B set IS the stalled response)
      - any path that used fallback (no LLM mixture to honor)

    Otherwise True iff any collapse / low-confidence signal fires:
      - `top_bin_share >= DIVERSITY_COLLAPSE_TOP_BIN (0.50)`  (mode collapse)
      - `panel_ready_bins_covered < K/2`   (panel shallow; only when panel_live)
      - `supervisor_confidence < CONFIDENT_MIXTURE_THRESHOLD`     (LLM unsure)
      - `recent_fallback_high`             (recent ticks fell back)

    BUG FIX (2026-05-28): the panel-coverage trigger is gated behind
    `panel_live`. PanelValue@K is deferred (doc §12) — `panel.select_panel`
    only runs at end-of-campaign consolidate and every parser hard-codes
    `panel_ready=False`, so in production `panel_ready_bins_covered` is a
    CONSTANT 0. With the trigger always live, `0 < K/2` fired every tick,
    forcing Category B on unconditionally and making the Option-C "LLM owns
    the allocation within stated bounds" design inert (a confident productive
    mixture was never honored within the wider Category A floors). Gating it
    behind `panel_live` (default False until the live panel selector is
    wired) restores LLM agency; the remaining genuine signals (collapse,
    low confidence, recent fallback, stalled) still trip Category B.
    """
    reasons: list[str] = []
    # Fallback path: no LLM mixture to respect, always clamp.
    if not supervisor_used:
        return True, ["fallback_path"]
    # Stalled always clamps (the clamps ARE the stalled response). deep_stall is
    # deliberately NOT here: the archive analysis showed the LLM reacts correctly
    # under collapse (targets high explore) — the escape blocker was feasibility,
    # fixed by the chain throttle + the Cat-A explore floor — so we preserve the
    # confident LLM's mixture and only force Cat-B when collapse/low-conf is
    # corroborated by the triggers below (which on a real deep_stall usually fire).
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
        # 2026-05-29: gate the panel-coverage trigger behind panel_live
        # (consistent with should_apply_category_b_clamps). PanelValue@K is
        # deferred so panel_ready_bins_covered is constant 0 → without the gate
        # `bin_coverage_short` was ALWAYS True, loosening the productive clamps
        # on every productive tick regardless of real structural collapse. Now
        # the loosening fires only on the genuine top_bin_share>=0.50 signal
        # (or shallow panel coverage once a live panel selector is wired).
        bin_coverage_short = panel_live and panel_ready_bins_covered < panel_size_K // 2
        if collapse_share >= DIVERSITY_COLLAPSE_TOP_BIN or bin_coverage_short:
            # FORCE a diversity pivot: raise the explore FLOOR (not just loosen the
            # cap) so the controller moves budget OFF the collapsing basin instead
            # of merely being allowed to. exploit_max 0.60 (was 0.65) so the
            # explore floor 0.20 + base rescue floor 0.20 is FEASIBLE (0.60+0.20+
            # 0.20=1.0); a 0.65 cap left explore renormalized below its floor.
            # exploit stays the majority lane so it still mints SU. explore cap ->0.40.
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
            explore_max=0.50,  # was 0.30
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
    """Apply per-state clamps with proper feasibility-preserving normalization.

    Algorithm: pin clamped (floor or ceiling) entries to their bounds,
    then distribute remaining mass proportionally over the unpinned
    entries. Iterates until fixed-point or max 5 passes to handle the
    rare case where renormalization pushes another entry across its
    bound.

    Diversity-aware loosening (user Q): when top_bin_share signals
    structural collapse on a productive run, clamps loosen to give
    the LLM headroom to pivot toward diversity (without removing
    the safety floors).

    §22.8.4: when `category_b_enabled=False` use `CATEGORY_A_CLAMPS`
    (bug+budget safety only) so the LLM's supervisor mixture is honored
    within minimal floors. The Selector gates this via
    `should_apply_category_b_clamps`.

    Returns (clamped_mixture, applied_clamps_log).
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
        # §22.8.4: Category-A-only path. No diversity-adjusted loosening
        # needed — there's no Category B ceiling to loosen.
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

    # First pass: clamp any out-of-bound input directly. This produces
    # the initial pin set.
    #
    # BUGFIX (2026-05-28): normalize to sum=1 BEFORE comparing against the
    # bounds. The Supervisor prompt tells the LLM its mode_mixture "will be
    # renormalized to sum to 1", so the LLM emits unnormalized values
    # (observed sums of 13, 100, … in 22-40% of production ticks). The
    # bounds (e.g. exploit_max=0.80) are fractions, so comparing a raw 40
    # against 0.80 pinned the mode to its ceiling regardless of the LLM's
    # actual intent — e.g. {40,20,40} (= intended {0.4,0.2,0.4}) was
    # mangled to {0.8,0.0,0.2} instead of the correct {0.45,0.35,0.20}.
    # DEFAULT_MIXTURES already sum to 1, so this is a no-op for the
    # fallback path.
    _clipped = {k: max(0.0, v) for k, v in mixture.items()}
    _tot = sum(_clipped.values())
    out = {k: v / _tot for k, v in _clipped.items()} if _tot > 0 else _clipped
    # Total over the FULL mode domain so the later sum(out[m] ...) over free modes
    # can't KeyError on a mode-incomplete input (review-found latent defect;
    # current callers always pass all 3 keys, so this is defensive).
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

        # Bug fix (edge case 5): all modes pinned at MINs that sum to < 1.0
        # used to return a non-normalized distribution (e.g. {0.45, 0.10, 0.05}
        # summing to 0.60). Pure mathematical floors can't be over-satisfied
        # without violating mins — but practically we want sum=1.0 so the
        # downstream largest_remainder rounding works. Scale ALL pinned
        # proportionally and re-check against each mode's max bound.
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

    # Smoke 8715062 finding: a feasible mode with a small but non-trivial
    # original mixture weight (e.g. explore=0.1) can end up with zero
    # quota after redistribute because largest-remainder favors the
    # bigger fractional. Protect it: any feasible mode with original
    # mixture >= MIN_REPRESENTATION_FLOOR that ends up at 0 quota steals
    # 1 slot from the largest-quota feasible mode (must have >1).
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
