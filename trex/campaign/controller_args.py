"""Legacy ``trex-controller`` argument adapter.

The recommended YAML interface translates to these arguments. Keeping this
parser separate makes the production controller consume resolved input instead
of also owning user-interface definitions.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path


QUOTA_REALIZATIONS = (
    "fractional_carry",
    "deterministic_deficit",
    "largest_remainder",
    "stochastic",
)


def controller_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="T-ReX asynchronous closed-loop controller"
    )
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument(
        "--target-constraint",
        type=Path,
        required=True,
        help="JSON file containing one TargetConstraint",
    )
    parser.add_argument("--target-pdb", required=True)
    parser.add_argument(
        "--vllm-base-url",
        default="http://127.0.0.1:12000/v1",
        help="OpenAI-compatible URL used by Planner and Supervisor",
    )
    parser.add_argument(
        "--llm-model",
        default="vllm/Qwen/Qwen3.6-27B-FP8",
        help="served model identifier used by Planner and Supervisor",
    )
    parser.add_argument(
        "--max-wall-h", type=float, default=48.0,
        help=(
            "total controller hours, including resumed elapsed time (default: 48); "
            "not a scheduler allocation or a hard per-start deadline"
        ),
    )
    parser.add_argument(
        "--worker-gpus",
        default=os.environ.get("TREX_WORKER_GPUS", "1,2,3"),
        help="comma-separated unique GPU identifiers for scientific workers",
    )
    parser.add_argument(
        "--enabled-families",
        default="",
        help=(
            "comma-separated family whitelist; empty enables every registered "
            "family"
        ),
    )
    parser.add_argument(
        "--enable-critic", type=int, choices=(0, 1), default=1,
        help="1 enables the deterministic flag-only Critic guard",
    )
    parser.add_argument(
        "--enable-evidence-skip", type=int, choices=(0, 1), default=0,
        help="1 reuses valid decisions when decision evidence is unchanged",
    )
    parser.add_argument(
        "--enable-exemplars", type=int, choices=(0, 1), default=1,
        help="1 includes best and near-miss exemplars in Planner evidence",
    )
    parser.add_argument(
        "--foldseek-su-tm-score",
        type=float,
        default=0.60,
        help="binder-chain Foldseek threshold for the live SU objective",
    )
    parser.add_argument(
        "--foldseek-collapse-tm-score",
        type=float,
        default=0.60,
        help="Foldseek threshold for collapse and near-miss diagnostics",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="campaign seed mixed into every seedable backend launch",
    )
    parser.add_argument(
        "--selector-quota-realization",
        choices=QUOTA_REALIZATIONS,
        default="fractional_carry",
    )
    parser.add_argument(
        "--selector-mode-window-k",
        type=int,
        default=10,
        help="window length for deterministic-deficit realization",
    )
    parser.add_argument(
        "--selector-adaptive-mode-window-k",
        type=int,
        choices=(0, 1),
        default=1,
        help="1 shortens the deterministic-deficit window in urgent states",
    )
    return parser


def parse_controller_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse and validate the execution boundary consumed by the controller."""

    args = controller_parser().parse_args(argv)
    for name, value in (
        ("--foldseek-su-tm-score", args.foldseek_su_tm_score),
        ("--foldseek-collapse-tm-score", args.foldseek_collapse_tm_score),
    ):
        if not 0.0 < value <= 1.0:
            raise SystemExit(f"{name} must be in (0, 1], got {value}")
    if not math.isfinite(args.max_wall_h) or args.max_wall_h <= 0:
        raise SystemExit(
            f"--max-wall-h must be finite and greater than zero, got {args.max_wall_h}"
        )
    if args.seed < 0:
        raise SystemExit(f"--seed must be >=0, got {args.seed}")
    if args.selector_mode_window_k < 1:
        raise SystemExit(
            "--selector-mode-window-k must be >=1, "
            f"got {args.selector_mode_window_k}"
        )
    return args
