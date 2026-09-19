"""Tests for the controller's isolated execution-input adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

from trex.campaign.controller_args import parse_controller_args


def _required(tmp_path: Path) -> list[str]:
    return [
        "--archive-root", str(tmp_path / "archive"),
        "--target-constraint", str(tmp_path / "target.json"),
        "--target-pdb", str(tmp_path / "target.pdb"),
    ]


def test_controller_args_expose_worker_and_llm_identity(tmp_path: Path) -> None:
    args = parse_controller_args([
        *_required(tmp_path),
        "--worker-gpus", "2,4",
        "--llm-model", "served/test-model",
    ])

    assert args.worker_gpus == "2,4"
    assert args.llm_model == "served/test-model"
    assert args.archive_root == tmp_path / "archive"


@pytest.mark.parametrize(
    "option,value",
    [
        ("--foldseek-su-tm-score", "0"),
        ("--foldseek-collapse-tm-score", "1.1"),
        ("--max-wall-h", "0"),
        ("--max-wall-h", "nan"),
        ("--max-wall-h", "inf"),
        ("--max-wall-h", "1e400"),
        ("--seed", "-1"),
        ("--selector-mode-window-k", "0"),
    ],
)
def test_controller_args_reject_invalid_values(
    tmp_path: Path, option: str, value: str,
) -> None:
    with pytest.raises(SystemExit):
        parse_controller_args([*_required(tmp_path), option, value])
