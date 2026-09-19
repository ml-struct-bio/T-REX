"""User-facing campaign configuration and inspection APIs.

The scientific controller remains in :mod:`trex.controller`.  This package
owns the boundary between a user-supplied campaign file and that controller:
loading, validation, target resolution, preflight checks, and concise archive
status output.
"""

from .config import ConfigError, load_campaign
from .models import (
    BackendPaths,
    CampaignConfig,
    MemoryConfig,
    ResolvedCampaign,
)
from .provenance import CampaignProvenanceArtifact, write_campaign_provenance
from .runtime import RuntimePaths

__all__ = [
    "BackendPaths",
    "CampaignConfig",
    "ConfigError",
    "MemoryConfig",
    "ResolvedCampaign",
    "CampaignProvenanceArtifact",
    "RuntimePaths",
    "load_campaign",
    "write_campaign_provenance",
]
