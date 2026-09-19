"""Per-family output parsers — convert worker output dirs → ResultRecord.

Each parser is a pure function `(output_dir: Path, ctx: ParserContext)
-> list[ResultRecord]`. Used by `phase5_launcher.py` (§22.1) to ingest
completed worker results into the T-ReX archive.

Diagnostic metrics (§4.1) are populated when present in the source
output; absent metrics are simply omitted from the ResultRecord.metrics
dict (NOT set to None — None means "this is a known unfilled slot"
which has a different semantics).
"""

from .types import ParseError, ParserContext

__all__ = ["ParseError", "ParserContext"]
