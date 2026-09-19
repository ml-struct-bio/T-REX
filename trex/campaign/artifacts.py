"""Write immutable, user-inspectable campaign input artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .models import ResolvedCampaign


INPUT_ARTIFACT_NAME = "campaign_input.yaml"
RESOLVED_ARTIFACT_NAME = "campaign_resolved.json"


@dataclass(frozen=True)
class CampaignArtifacts:
    input_path: Path
    resolved_path: Path


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _available(path: Path, content: bytes) -> bool:
    return not path.exists() or path.read_bytes() == content


def _paired_paths(
    root: Path, input_content: bytes, resolved_content: bytes,
) -> tuple[Path, Path]:
    input_path = root / INPUT_ARTIFACT_NAME
    resolved_path = root / RESOLVED_ARTIFACT_NAME
    if _available(input_path, input_content) and _available(
        resolved_path, resolved_content
    ):
        return input_path, resolved_path

    digest = hashlib.sha256(
        input_content + b"\0trex-resolved\0" + resolved_content
    ).hexdigest()
    for length in (12, 24, 64):
        suffix = digest[:length]
        input_candidate = input_path.with_name(
            f"{input_path.stem}_{suffix}{input_path.suffix}"
        )
        resolved_candidate = resolved_path.with_name(
            f"{resolved_path.stem}_{suffix}{resolved_path.suffix}"
        )
        if _available(input_candidate, input_content) and _available(
            resolved_candidate, resolved_content
        ):
            return input_candidate, resolved_candidate
    raise RuntimeError(f"cannot choose non-overwriting artifact paths under {root}")


def write_campaign_artifacts(campaign: ResolvedCampaign) -> CampaignArtifacts:
    """Persist exact input and resolved input without overwriting prior runs."""

    root = campaign.config.run.archive_root
    source_content = campaign.source_content
    resolved_content = (
        json.dumps(campaign.as_dict(), indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    input_path, resolved_path = _paired_paths(
        root, source_content, resolved_content
    )
    _atomic_write(input_path, source_content)
    _atomic_write(resolved_path, resolved_content)
    return CampaignArtifacts(input_path=input_path, resolved_path=resolved_path)
