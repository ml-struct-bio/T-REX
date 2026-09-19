#!/usr/bin/env python3
"""Render the source-backed available LLM prompt catalog.

The checked-in snapshot is the canonical no-cross-campaign-memory prompt set.
Pass ``--use-configured-memory`` to inspect the exact runtime extension selected
by ``TREX_CROSS_CAMPAIGN_MEMORY``; run provenance records that file's SHA256.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

use_configured_memory = "--use-configured-memory" in sys.argv
if use_configured_memory:
    sys.argv.remove("--use-configured-memory")
else:
    os.environ.pop("TREX_CROSS_CAMPAIGN_MEMORY", None)

from trex.prompt_catalog import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
