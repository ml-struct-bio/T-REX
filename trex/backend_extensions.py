"""Operator-installed backend adapters for T-REX.

Adapters extend the deterministic action surface; they are trusted Python code
installed by the operator, never code or shell text supplied by the LLM. The
built-in publication backends continue to use their frozen controller paths.
Third-party adapters are deliberately diagnostic-only and must pass through
the canonical score-conversion route before they can receive strict/SU credit.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import inspect
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .capability_registry import Capability
from .output_parsers import ParserContext
from .schemas import ActionCandidate, ResultRecord, TargetConstraint


BACKEND_ENTRY_POINT_GROUP = "trex.backends"
_FAMILY_RE = re.compile(r"[a-z][a-z0-9_]{1,63}")
_CACHE_KEY: tuple[str, str] | None = None
_CACHE: dict[str, "BackendAdapter"] | None = None


@dataclass(frozen=True)
class BackendLaunchContext:
    """Validated controller context available to an adapter command builder."""

    candidate: ActionCandidate
    output_root: Path
    target: TargetConstraint
    target_pdb: Path
    gpu_id: str
    length_range: tuple[int, int]
    parent_pdb: Path | None = None
    parent_result_id: str = ""
    target_chains_csv: str = ""
    binder_chain: str = ""


@dataclass(frozen=True)
class BackendCommand:
    """A direct argv launch; shell command strings are intentionally unsupported."""

    argv: Sequence[str]
    output_dir: Path
    cwd: Path | None = None
    env: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class BackendAdapter:
    """Complete extension contract for one diagnostic backend family."""

    capability: Capability
    build_command: Callable[[BackendLaunchContext], BackendCommand]
    parse_output: Callable[[Path, ParserContext], list[ResultRecord]]
    budget_parameters: tuple[str, ...]
    budget_defaults: Mapping[str, int]
    eval_budget_cap: int
    hard_ceiling_seconds: int = 7200

    @property
    def family(self) -> str:
        return self.capability.family

    @property
    def budget_formula(self) -> str:
        return " * ".join(self.budget_parameters)

    def estimate_budget(self, config_delta: Mapping[str, object] | None) -> int:
        values = config_delta or {}
        budget = 1
        for name in self.budget_parameters:
            default = int(self.budget_defaults[name])
            raw = values.get(name, default)
            value = int(raw) if isinstance(raw, (int, float)) else default
            budget *= max(1, value)
        return budget


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_backend_adapter(adapter: BackendAdapter) -> None:
    """Fail closed on incomplete or unsafe extension metadata."""

    if not isinstance(adapter, BackendAdapter):
        raise TypeError(f"backend provider returned {type(adapter).__name__}, expected BackendAdapter")
    cap = adapter.capability
    if not _FAMILY_RE.fullmatch(cap.family):
        raise ValueError(f"invalid backend family name: {cap.family!r}")
    if cap.role not in {"generator", "seq_redesign"}:
        raise ValueError(
            f"extension {cap.family} role must be generator or seq_redesign; "
            "new canonical scorers require a reviewed core integration"
        )
    if not cap.outputs_diagnostic_only:
        raise ValueError(
            f"extension {cap.family} must set outputs_diagnostic_only=True; "
            "extension-native scores cannot mint strict/SU credit"
        )
    if cap.availability != "available":
        raise ValueError(f"extension {cap.family} must register as available")
    if not callable(adapter.build_command) or not callable(adapter.parse_output):
        raise ValueError(f"extension {cap.family} needs command and parser callables")
    if not adapter.budget_parameters or len(set(adapter.budget_parameters)) != len(adapter.budget_parameters):
        raise ValueError(f"extension {cap.family} needs unique budget parameters")
    if adapter.eval_budget_cap < 1:
        raise ValueError(f"extension {cap.family} eval_budget_cap must be positive")
    if not 60 <= adapter.hard_ceiling_seconds <= 7 * 24 * 3600:
        raise ValueError(f"extension {cap.family} hard ceiling must be 60s to 7d")
    for name in adapter.budget_parameters:
        if name not in adapter.budget_defaults:
            raise ValueError(f"extension {cap.family} has no default for budget parameter {name}")
        default = adapter.budget_defaults[name]
        if not isinstance(default, int) or isinstance(default, bool) or default < 1:
            raise ValueError(f"extension {cap.family} default {name} must be a positive integer")
        allowed = cap.allowed_params.get(name)
        if not (isinstance(allowed, tuple) and len(allowed) == 2):
            raise ValueError(
                f"extension {cap.family} budget parameter {name} needs a numeric allowed range"
            )
        if not float(allowed[0]) <= default <= float(allowed[1]):
            raise ValueError(f"extension {cap.family} default {name} is outside its allowed range")
    default_budget = adapter.estimate_budget({})
    if default_budget > adapter.eval_budget_cap:
        raise ValueError(
            f"extension {cap.family} default budget {default_budget} exceeds cap "
            f"{adapter.eval_budget_cap}"
        )


def _coerce_provider(value: Any, provider: str) -> BackendAdapter:
    if isinstance(value, BackendAdapter):
        adapter = value
    elif callable(value):
        adapter = value()
    else:
        raise TypeError(f"backend provider {provider} is not an adapter or zero-argument factory")
    validate_backend_adapter(adapter)
    return adapter


def _load_module_provider(spec: str) -> BackendAdapter:
    module_name, separator, attribute = spec.strip().partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError(
            f"invalid TREX_BACKEND_PLUGINS entry {spec!r}; use module:attribute"
        )
    module = importlib.import_module(module_name)
    try:
        value = getattr(module, attribute)
    except AttributeError as exc:
        raise ValueError(f"backend provider not found: {spec}") from exc
    return _coerce_provider(value, spec)


def _entry_points() -> list[Any]:
    if os.environ.get("TREX_DISABLE_ENTRYPOINT_BACKENDS", "0") == "1":
        return []
    points = importlib.metadata.entry_points()
    if hasattr(points, "select"):
        return list(points.select(group=BACKEND_ENTRY_POINT_GROUP))
    return list(points.get(BACKEND_ENTRY_POINT_GROUP, ()))  # pragma: no cover


def load_backend_adapters(*, refresh: bool = False) -> dict[str, BackendAdapter]:
    """Load adapters from entry points and explicit ``module:attribute`` specs."""

    global _CACHE_KEY, _CACHE
    explicit = os.environ.get("TREX_BACKEND_PLUGINS", "").strip()
    disabled = os.environ.get("TREX_DISABLE_ENTRYPOINT_BACKENDS", "0").strip()
    key = (explicit, disabled)
    if not refresh and _CACHE is not None and _CACHE_KEY == key:
        return dict(_CACHE)

    providers: list[tuple[str, Any]] = []
    for point in sorted(_entry_points(), key=lambda item: (item.name, item.value)):
        providers.append((f"entrypoint:{point.name}={point.value}", point.load()))
    for spec in (item.strip() for item in explicit.split(",")):
        if spec:
            providers.append((f"env:{spec}", _load_module_provider(spec)))

    adapters: dict[str, BackendAdapter] = {}
    for provider, value in providers:
        adapter = _coerce_provider(value, provider)
        if adapter.family in adapters:
            raise ValueError(f"duplicate backend family {adapter.family!r} from {provider}")
        adapters[adapter.family] = adapter

    _CACHE_KEY = key
    _CACHE = dict(adapters)
    return adapters


def clear_backend_adapter_cache() -> None:
    """Clear loader state (primarily for tests and interactive development)."""

    global _CACHE_KEY, _CACHE
    _CACHE_KEY = None
    _CACHE = None


def get_backend_adapter(family: str) -> BackendAdapter | None:
    return load_backend_adapters().get(family)


def install_extension_budget_metadata(adapters: Mapping[str, BackendAdapter]) -> None:
    """Expose adapter work-unit bounds to the existing deterministic guardrail."""

    from . import capability_registry as registry

    for family, adapter in adapters.items():
        registry.EVAL_BUDGET_CAP_PER_FAMILY[family] = adapter.eval_budget_cap
        registry.EVAL_BUDGET_DEFAULTS_PER_FAMILY[family] = dict(adapter.budget_defaults)
        registry.EVAL_BUDGET_FORMULA_PER_FAMILY[family] = adapter.budget_formula


def validate_backend_command(command: BackendCommand, output_root: Path) -> BackendCommand:
    """Validate an adapter-built direct command before a worker is started."""

    argv = tuple(str(item) for item in command.argv)
    if not argv or any(not item or "\x00" in item for item in argv):
        raise ValueError("backend command argv must contain non-empty strings")
    root = output_root.expanduser().resolve()
    out = command.output_dir.expanduser()
    if not out.is_absolute():
        out = root / out
    out = out.resolve()
    try:
        out.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"backend output_dir must stay under {root}: {out}") from exc
    cwd = command.cwd.expanduser().resolve() if command.cwd is not None else None
    if cwd is not None and not cwd.is_dir():
        raise ValueError(f"backend cwd does not exist: {cwd}")
    env = {str(key): str(value) for key, value in command.env.items()}
    if any(not key or "=" in key or "\x00" in key or "\x00" in value for key, value in env.items()):
        raise ValueError("backend environment contains an invalid name or value")
    return BackendCommand(argv=argv, output_dir=out, cwd=cwd, env=env)


def validate_extension_records(
    adapter: BackendAdapter,
    records: list[ResultRecord],
    context: ParserContext,
) -> list[ResultRecord]:
    """Enforce provenance and diagnostic-only semantics on plugin parser rows."""

    seen: set[str] = set()
    for record in records:
        if not isinstance(record, ResultRecord):
            raise ValueError(f"extension {adapter.family} parser returned a non-ResultRecord")
        if record.backend_family != adapter.family:
            raise ValueError(
                f"extension {adapter.family} emitted backend_family={record.backend_family!r}"
            )
        if record.target_id != context.target_id:
            raise ValueError(
                f"extension {adapter.family} emitted target_id={record.target_id!r}, "
                f"expected {context.target_id!r}"
            )
        if not record.result_id or record.result_id in seen:
            raise ValueError(f"extension {adapter.family} emitted an empty or duplicate result_id")
        seen.add(record.result_id)
        canonical = {"pLDDT", "iPAE", "binder_scRMSD"} & set(record.metrics or {})
        canonical |= {"pLDDT", "iPAE", "binder_scRMSD"} & set(record.metrics_calibrated or {})
        if canonical:
            raise ValueError(
                f"diagnostic extension {adapter.family} emitted canonical strict fields "
                f"{sorted(canonical)}; use backend-prefixed diagnostics and canonical score conversion"
            )
    return records


def backend_extension_provenance() -> list[dict[str, Any]]:
    """Return compact provider/source metadata for ``run_provenance.json``."""

    rows: list[dict[str, Any]] = []
    for family, adapter in sorted(load_backend_adapters().items()):
        callback = adapter.build_command
        module = inspect.getmodule(callback)
        source = inspect.getsourcefile(callback)
        source_path = Path(source).resolve() if source else None
        rows.append({
            "family": family,
            "module": module.__name__ if module is not None else None,
            "source_path": str(source_path) if source_path else None,
            "source_sha256": _sha256(source_path) if source_path and source_path.is_file() else None,
            "budget_formula": adapter.budget_formula,
            "budget_defaults": dict(adapter.budget_defaults),
            "eval_budget_cap": adapter.eval_budget_cap,
            "hard_ceiling_seconds": adapter.hard_ceiling_seconds,
            "diagnostic_only": True,
        })
    return rows


__all__ = [
    "BACKEND_ENTRY_POINT_GROUP",
    "BackendAdapter",
    "BackendCommand",
    "BackendLaunchContext",
    "backend_extension_provenance",
    "clear_backend_adapter_cache",
    "get_backend_adapter",
    "install_extension_budget_metadata",
    "load_backend_adapters",
    "validate_backend_adapter",
    "validate_backend_command",
    "validate_extension_records",
]
