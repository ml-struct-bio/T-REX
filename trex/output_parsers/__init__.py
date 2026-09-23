"""Convert backend output directories and ParserContext into ResultRecords.

Preserve available source diagnostics and omit measurements that were not reported.
"""

from .types import ParseError, ParserContext

__all__ = ["ParseError", "ParserContext"]
