#!/usr/bin/env python3
"""Compatibility wrapper for ``python -m trex.validation``."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from trex.validation import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
