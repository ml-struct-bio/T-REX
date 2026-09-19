"""Compatibility environment for the current backend adapters."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator, Mapping


_SENSITIVE_NAME_FRAGMENTS = (
    "CREDENTIAL",
    "PASSWORD",
    "SECRET",
    "TOKEN",
)


def inspectable_environment(
    values: Mapping[str, str],
    *,
    prefix: str,
    excluded_names: set[str] | frozenset[str] = frozenset(),
) -> dict[str, str]:
    """Return a stable environment snapshot without persisting credentials."""

    snapshot: dict[str, str] = {}
    for name, value in sorted(values.items()):
        if not name.startswith(prefix) or name in excluded_names:
            continue
        upper_name = name.upper()
        sensitive = upper_name.endswith("_KEY") or any(
            fragment in upper_name for fragment in _SENSITIVE_NAME_FRAGMENTS
        )
        snapshot[name] = "<redacted>" if sensitive else value
    return snapshot


@contextmanager
def applied_environment(values: Mapping[str, str]) -> Iterator[None]:
    """Temporarily apply explicit campaign values and restore the process env."""

    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
