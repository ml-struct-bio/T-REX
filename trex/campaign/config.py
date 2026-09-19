"""Load and validate the single-file T-ReX campaign input contract."""

from __future__ import annotations

import hashlib
import math
import os
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml
from yaml.constructor import ConstructorError

from ..capability_registry import default_registry
from ..resource_paths import publication_data_path
from ..targets import resolve_target
from ..validation import DEFAULT_FAMILIES
from .models import (
    CAMPAIGN_SCHEMA_VERSION,
    BackendPaths,
    CampaignConfig,
    LLMConfig,
    MemoryConfig,
    PolicyConfig,
    ResolvedCampaign,
    RunConfig,
    TargetInput,
)
from .runtime.paths import (
    BACKEND_PATH_ENVIRONMENT_VARIABLES,
    EXECUTABLE_BACKEND_PATH_FIELDS,
)


class ConfigError(ValueError):
    """A campaign file is invalid or cannot be resolved."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects visually ambiguous duplicate keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


_TOP_LEVEL_KEYS = {
    "schema_version",
    "name",
    "target",
    "run",
    "llm",
    "policy",
    "memory",
    "backends",
}
_TARGET_KEYS = {"name", "asset_root", "constraint", "pdb"}
_RUN_KEYS = {
    "archive_root",
    "max_wall_hours",
    "seed",
    "worker_gpus",
    "enabled_families",
}
_LLM_KEYS = {"base_url", "model"}
_MEMORY_KEYS = {"cross_campaign_path"}
_POLICY_KEYS = {
    "critic",
    "evidence_skip",
    "exemplars",
    "foldseek_su_tm_score",
    "foldseek_collapse_tm_score",
    "selector_quota_realization",
    "selector_mode_window_k",
    "selector_adaptive_mode_window_k",
}
_BACKEND_KEYS = set(BackendPaths.__dataclass_fields__)
_QUOTA_REALIZATIONS = {
    "fractional_carry",
    "deterministic_deficit",
    "largest_remainder",
    "stochastic",
}


def _absolute_path(path: Path, *, dereference: bool = True) -> Path:
    """Make a path absolute while preserving virtualenv executable symlinks."""

    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = Path(os.path.abspath(expanded))
    return expanded.resolve() if dereference else expanded


def _object(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{location} must be a mapping")
    return value


def _reject_unknown(data: Mapping[str, Any], allowed: set[str], location: str) -> None:
    unknown = sorted(str(key) for key in data if key not in allowed)
    if unknown:
        raise ConfigError(
            f"{location} contains unknown key(s): {', '.join(unknown)}; "
            f"allowed: {', '.join(sorted(allowed))}"
        )


def _required_text(data: Mapping[str, Any], key: str, location: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{location}.{key} must be a non-empty string")
    return value.strip()


def _text(data: Mapping[str, Any], key: str, default: str, location: str) -> str:
    value = data.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{location}.{key} must be a non-empty string")
    return value.strip()


def _boolean(data: Mapping[str, Any], key: str, default: bool, location: str) -> bool:
    value = data.get(key, default)
    if type(value) is not bool:
        raise ConfigError(f"{location}.{key} must be true or false")
    return value


def _integer(data: Mapping[str, Any], key: str, default: int, location: str) -> int:
    value = data.get(key, default)
    if type(value) is not int:
        raise ConfigError(f"{location}.{key} must be an integer")
    return value


def _number(data: Mapping[str, Any], key: str, default: float, location: str) -> float:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{location}.{key} must be a number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ConfigError(f"{location}.{key} must be finite") from exc
    if not math.isfinite(number):
        raise ConfigError(f"{location}.{key} must be finite")
    return number


def _path_value(
    data: Mapping[str, Any],
    key: str,
    base_dir: Path,
    location: str,
    *,
    required: bool = False,
    dereference: bool = True,
) -> Path | None:
    value = data.get(key)
    if value is None:
        if required:
            raise ConfigError(f"{location}.{key} is required")
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{location}.{key} must be a path string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return _absolute_path(path, dereference=dereference)


def _string_tuple(
    data: Mapping[str, Any],
    key: str,
    default: Iterable[str],
    location: str,
) -> tuple[str, ...]:
    value = data.get(key, list(default))
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{location}.{key} must be a non-empty list")
    items: list[str] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (str, int)):
            raise ConfigError(f"{location}.{key}[{index}] must be a string or integer")
        text = str(item).strip()
        if not text:
            raise ConfigError(f"{location}.{key}[{index}] must not be empty")
        if "," in text or "\n" in text or "\r" in text:
            raise ConfigError(f"{location}.{key}[{index}] must be one unambiguous item")
        items.append(text)
    if len(items) != len(set(items)):
        raise ConfigError(f"{location}.{key} must not contain duplicates")
    return tuple(items)


def _load_yaml(path: Path, source_content: bytes | None = None) -> Mapping[str, Any]:
    if source_content is None:
        try:
            source_content = path.read_bytes()
        except OSError as exc:
            raise ConfigError(f"cannot read campaign config {path}: {exc}") from exc
    try:
        payload = yaml.load(source_content, Loader=_UniqueKeyLoader) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    return _object(payload, "campaign config")


def parse_campaign(
    path: Path | str,
    *,
    _source_content: bytes | None = None,
) -> CampaignConfig:
    """Parse one YAML file into a strictly validated typed input model."""

    source = Path(path).expanduser().resolve()
    payload = _load_yaml(source, _source_content)
    _reject_unknown(payload, _TOP_LEVEL_KEYS, "campaign config")
    schema = payload.get("schema_version")
    if schema != CAMPAIGN_SCHEMA_VERSION:
        raise ConfigError(
            f"campaign config.schema_version must be {CAMPAIGN_SCHEMA_VERSION!r}, "
            f"got {schema!r}"
        )
    name = _text(payload, "name", source.stem, "campaign config")
    base_dir = source.parent

    target_data = _object(payload.get("target"), "target")
    _reject_unknown(target_data, _TARGET_KEYS, "target")
    target = TargetInput(
        name=_required_text(target_data, "name", "target"),
        asset_root=_path_value(target_data, "asset_root", base_dir, "target"),
        constraint=_path_value(target_data, "constraint", base_dir, "target"),
        pdb=_path_value(target_data, "pdb", base_dir, "target"),
    )

    run_data = _object(payload.get("run"), "run")
    _reject_unknown(run_data, _RUN_KEYS, "run")
    max_wall_hours = _number(
        run_data, "max_wall_hours", RunConfig.max_wall_hours, "run"
    )
    seed = _integer(run_data, "seed", 0, "run")
    worker_gpus = _string_tuple(run_data, "worker_gpus", ("1", "2", "3"), "run")
    enabled_families = _string_tuple(
        run_data, "enabled_families", DEFAULT_FAMILIES, "run"
    )
    if max_wall_hours <= 0:
        raise ConfigError("run.max_wall_hours must be greater than zero")
    if seed < 0:
        raise ConfigError("run.seed must be non-negative")
    run = RunConfig(
        archive_root=_path_value(
            run_data, "archive_root", base_dir, "run", required=True
        ),  # type: ignore[arg-type]
        max_wall_hours=max_wall_hours,
        seed=seed,
        worker_gpus=worker_gpus,
        enabled_families=enabled_families,
    )

    llm_data = _object(payload.get("llm", {}), "llm")
    _reject_unknown(llm_data, _LLM_KEYS, "llm")
    llm = LLMConfig(
        base_url=_text(llm_data, "base_url", "http://127.0.0.1:12000/v1", "llm"),
        model=_text(llm_data, "model", "vllm/Qwen/Qwen3.6-27B-FP8", "llm"),
    )

    policy_data = _object(payload.get("policy", {}), "policy")
    _reject_unknown(policy_data, _POLICY_KEYS, "policy")
    su_tm = _number(policy_data, "foldseek_su_tm_score", 0.60, "policy")
    collapse_tm = _number(policy_data, "foldseek_collapse_tm_score", 0.60, "policy")
    if not 0 < su_tm <= 1:
        raise ConfigError("policy.foldseek_su_tm_score must be in (0, 1]")
    if not 0 < collapse_tm <= 1:
        raise ConfigError("policy.foldseek_collapse_tm_score must be in (0, 1]")
    quota = _text(
        policy_data, "selector_quota_realization", "fractional_carry", "policy"
    )
    if quota not in _QUOTA_REALIZATIONS:
        raise ConfigError(
            "policy.selector_quota_realization must be one of: "
            + ", ".join(sorted(_QUOTA_REALIZATIONS))
        )
    mode_window = _integer(
        policy_data, "selector_mode_window_k",
        PolicyConfig.selector_mode_window_k, "policy"
    )
    if mode_window < 1:
        raise ConfigError("policy.selector_mode_window_k must be at least 1")
    policy = PolicyConfig(
        critic=_boolean(policy_data, "critic", True, "policy"),
        evidence_skip=_boolean(policy_data, "evidence_skip", False, "policy"),
        exemplars=_boolean(policy_data, "exemplars", True, "policy"),
        foldseek_su_tm_score=su_tm,
        foldseek_collapse_tm_score=collapse_tm,
        selector_quota_realization=quota,
        selector_mode_window_k=mode_window,
        selector_adaptive_mode_window_k=_boolean(
            policy_data, "selector_adaptive_mode_window_k",
            PolicyConfig.selector_adaptive_mode_window_k, "policy"
        ),
    )

    memory_data = _object(payload.get("memory", {}), "memory")
    _reject_unknown(memory_data, _MEMORY_KEYS, "memory")
    memory = MemoryConfig(
        cross_campaign_path=_path_value(
            memory_data, "cross_campaign_path", base_dir, "memory"
        )
    )

    backend_data = _object(payload.get("backends", {}), "backends")
    _reject_unknown(backend_data, _BACKEND_KEYS, "backends")
    backends = BackendPaths(
        **{
            key: _path_value(
                backend_data,
                key,
                base_dir,
                "backends",
                dereference=key not in EXECUTABLE_BACKEND_PATH_FIELDS,
            )
            for key in sorted(_BACKEND_KEYS)
        }
    )
    return CampaignConfig(
        name=name,
        target=target,
        run=run,
        llm=llm,
        policy=policy,
        memory=memory,
        backends=backends,
    )


def _resolve_backend_environment(
    config: CampaignConfig,
) -> tuple[CampaignConfig, tuple[str, ...]]:
    """Materialize explicit, inherited, and derived backend execution paths."""

    supplied = vars(config.backends)
    notes: list[str] = []

    def choose(
        field_name: str,
        default: Path | None = None,
    ) -> Path | None:
        explicit = supplied[field_name]
        if explicit is not None:
            return explicit
        environment_name = BACKEND_PATH_ENVIRONMENT_VARIABLES[field_name]
        raw = os.environ.get(environment_name, "").strip()
        dereference = field_name not in EXECUTABLE_BACKEND_PATH_FIELDS
        if raw:
            notes.append(f"backends.{field_name} inherited from {environment_name}")
            return _absolute_path(Path(raw), dereference=dereference)
        if default is None:
            return None
        return _absolute_path(default, dereference=dereference)

    package_root = Path(__file__).resolve().parents[2]
    repo_root = choose("repo_root", package_root)
    assert repo_root is not None
    external_root = choose("external_root", repo_root / "external")
    assert external_root is not None
    complexa_repo = choose(
        "complexa_repo", repo_root / "external" / "Proteina-Complexa"
    )
    assert complexa_repo is not None
    legacy_complexa_repo = choose("legacy_complexa_repo", complexa_repo)
    bindcraft_repo = choose("bindcraft_repo", external_root / "BindCraft")
    assert bindcraft_repo is not None
    boltzgen_repo = choose("boltzgen_repo", external_root / "BoltzGen")
    foldseek = shutil.which("foldseek")
    mmseqs = shutil.which("mmseqs")
    values = {
        "repo_root": repo_root,
        "external_root": external_root,
        "complexa_repo": complexa_repo,
        "legacy_complexa_repo": legacy_complexa_repo,
        "complexa_python": choose(
            "complexa_python", complexa_repo / ".venv" / "bin" / "python"
        ),
        "bindcraft_repo": bindcraft_repo,
        "bindcraft_env": choose("bindcraft_env", bindcraft_repo / ".venv"),
        "boltzgen_repo": boltzgen_repo,
        "boltzgen_binary": choose(
            "boltzgen_binary",
            external_root / ".venvs" / "boltzgen" / "bin" / "boltzgen",
        ),
        "boltzgen_cache": choose(
            "boltzgen_cache", external_root / "checkpoints" / "boltzgen"
        ),
        "foldseek_binary": choose(
            "foldseek_binary", Path(foldseek) if foldseek else None
        ),
        "mmseqs_binary": choose("mmseqs_binary", Path(mmseqs) if mmseqs else None),
        "qwen_model_path": choose("qwen_model_path"),
        "qwen_model_manifest": choose(
            "qwen_model_manifest",
            publication_data_path(
                repo_root, "config/trex/qwen3_6_27b_fp8_model_manifest.json"
            ),
        ),
    }
    if config.target.asset_root is None:
        raw_asset_root = os.environ.get("TREX_TARGET_ASSET_ROOT", "").strip()
        if raw_asset_root:
            target = replace(
                config.target,
                asset_root=Path(raw_asset_root).expanduser().resolve(),
            )
            config = replace(config, target=target)
            notes.append("target.asset_root inherited from TREX_TARGET_ASSET_ROOT")
    return replace(config, backends=BackendPaths(**values)), tuple(notes)


def _effective_families(
    requested: tuple[str, ...]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    registry = default_registry()
    unknown = [family for family in requested if registry.get(family) is None]
    if unknown:
        raise ConfigError(
            f"run.enabled_families contains unknown family(s): {', '.join(unknown)}"
        )
    effective = list(requested)
    diagnostic = [
        family
        for family in requested
        if registry.get(family) is not None
        and registry.get(family).outputs_diagnostic_only  # type: ignore[union-attr]
    ]
    notes: list[str] = []
    if diagnostic and "structure_refilter" not in effective:
        effective.append("structure_refilter")
        notes.append(
            "auto-enabled structure_refilter because diagnostic-only families "
            f"need canonical AF2 scoring: {', '.join(diagnostic)}"
        )
    return tuple(effective), tuple(notes)


def load_campaign(path: Path | str) -> ResolvedCampaign:
    """Load, validate, and resolve a campaign file to concrete execution input."""

    source = Path(path).expanduser().resolve()
    try:
        source_content = source.read_bytes()
    except OSError as exc:
        raise ConfigError(f"cannot read campaign config {source}: {exc}") from exc
    parsed = parse_campaign(source, _source_content=source_content)
    config, environment_notes = _resolve_backend_environment(parsed)
    try:
        target = resolve_target(
            config.target.name,
            repo_root=config.backends.repo_root,
            asset_root=config.target.asset_root,
            target_config=config.target.constraint,
            target_pdb=config.target.pdb,
        )
        effective, notes = _effective_families(config.run.enabled_families)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    return ResolvedCampaign(
        config=config,
        target=target,
        source_path=source,
        source_content=source_content,
        source_sha256=hashlib.sha256(source_content).hexdigest(),
        effective_families=effective,
        resolution_notes=environment_notes + notes,
    )
