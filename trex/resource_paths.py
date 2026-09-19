"""Resolve immutable publication data in a checkout or installed wheel."""

from __future__ import annotations

from pathlib import Path


PACKAGED_DATA_ROOT = Path(__file__).resolve().parent / "data"


def publication_data_path(
    repository_root: Path | str,
    relative_path: Path | str,
) -> Path:
    """Prefer checkout data and fall back to the wheel's byte-identical copy."""

    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"publication data path must be relative: {relative}")
    checkout_path = Path(repository_root).expanduser().resolve() / relative
    if checkout_path.exists():
        return checkout_path
    packaged_path = PACKAGED_DATA_ROOT / relative
    return packaged_path if packaged_path.exists() else checkout_path
