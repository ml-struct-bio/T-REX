"""T-ReX archive: append-only JSONL, one file per record type.

Layout (under archive root):
  result_records.jsonl
  evidence_summaries.jsonl
  metric_calibrations.jsonl
  runtime_buckets.jsonl
  target_constraints.jsonl
  hypothesis_cards.jsonl
  action_candidates.jsonl
  supervisor_decisions.jsonl
  llm_call_records.jsonl
  launch_decisions.jsonl
  dispatch_records.jsonl
  route_records.jsonl
  panel_selections.jsonl

The writer is process-safe via O_APPEND + line-flush. Multiple workers
can append concurrently; readers should tolerate partial writes (skip
lines that fail JSON parse).
"""

from __future__ import annotations

import copy
import fcntl
import json
import os
import time
from dataclasses import fields, is_dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Iterator, get_args, get_origin, get_type_hints

from .schemas import (
    ActionCandidate,
    EvidenceSummary,
    HypothesisCard,
    LLMCallRecord,
    LaunchDecision,
    DispatchRecord,
    MetricCalibration,
    PanelSelection,
    ResultRecord,
    RouteRecord,
    RuntimeBucket,
    SupervisorDecision,
    TargetConstraint,
    to_jsonable,
)


# Record type → file name. MissingCandidateRequest entry removed
# 2026-05-26 PM along with the rest of the MCR machinery (plan §10.6).
RECORD_FILES: dict[type, str] = {
    ResultRecord: "result_records.jsonl",
    EvidenceSummary: "evidence_summaries.jsonl",
    MetricCalibration: "metric_calibrations.jsonl",
    RuntimeBucket: "runtime_buckets.jsonl",
    TargetConstraint: "target_constraints.jsonl",
    HypothesisCard: "hypothesis_cards.jsonl",
    ActionCandidate: "action_candidates.jsonl",
    SupervisorDecision: "supervisor_decisions.jsonl",
    LLMCallRecord: "llm_call_records.jsonl",
    LaunchDecision: "launch_decisions.jsonl",
    DispatchRecord: "dispatch_records.jsonl",
    RouteRecord: "route_records.jsonl",
    PanelSelection: "panel_selections.jsonl",
}


def _inner_dataclass(ftype_str: str) -> type | None:
    """Parse a string annotation and return the inner dataclass type from
    `trex.schemas`, or None if not a known dataclass.

    Recognized forms (2026-05-26 expanded — was list-only):
      • ``list[Dataclass]``           → returns Dataclass
      • ``dict[str, Dataclass]``      → returns Dataclass (value type)
      • ``Dataclass | None``          → returns Dataclass
      • ``Dataclass``                 → returns Dataclass

    Python 3.9's `get_type_hints` chokes on PEP-604 unions in this codebase
    (`X | None`), so we fall back to lightweight string matching on the
    raw `dataclasses.Field.type` annotation.
    """
    if not isinstance(ftype_str, str):
        return None
    from . import schemas as _schemas

    def _resolve(name: str) -> type | None:
        cand = getattr(_schemas, name.strip(), None)
        if isinstance(cand, type) and is_dataclass(cand):
            return cand
        return None

    s = ftype_str.strip()
    # strip outer Optional / "| None" wrappers
    if "|" in s and not s.startswith("list[") and not s.startswith("dict["):
        first = s.split("|")[0].strip()
        res = _resolve(first)
        if res is not None:
            return res

    if s.startswith("list[") and s.endswith("]"):
        inner = s[len("list[") : -1].strip()
        if "|" in inner:
            inner = inner.split("|")[0].strip()
        return _resolve(inner)

    if s.startswith("dict[") and s.endswith("]"):
        inner = s[len("dict[") : -1].strip()
        # dict[K, V] — extract V (last comma at top level)
        depth = 0
        comma_idx = -1
        for i, ch in enumerate(inner):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
            elif ch == "," and depth == 0:
                comma_idx = i
        if comma_idx >= 0:
            v = inner[comma_idx + 1 :].strip()
            if "|" in v:
                v = v.split("|")[0].strip()
            return _resolve(v)

    # Bare dataclass name
    return _resolve(s)


def _wrapper_kind(ftype_str: str) -> str:
    """Return 'list', 'dict', or 'single' to disambiguate empty containers.

    F-001 fix (2026-05-26): when ``raw == {}`` and the field is
    ``dict[str, Dataclass]``, we must emit ``{}`` (empty mapping) rather
    than treat ``{}`` as a single Dataclass instance (which then fails the
    Dataclass constructor with missing-required-arg errors). The earlier
    code distinguished cases solely by inspecting ``raw`` contents — that
    is ambiguous for empty dicts. The annotation string carries the
    information; this helper extracts it.
    """
    if not isinstance(ftype_str, str):
        return "single"
    s = ftype_str.strip()
    if s.startswith("list["):
        return "list"
    if s.startswith("dict["):
        return "dict"
    return "single"


@lru_cache(maxsize=None)
def _reconstruction_plan(cls: type) -> tuple[tuple[str, type | None, str], ...]:
    """Cached nested-dataclass metadata for hot archive reads."""
    if not is_dataclass(cls):
        return ()
    return tuple(
        (fld.name, _inner_dataclass(fld.type), _wrapper_kind(fld.type))
        for fld in fields(cls)
    )


def _reconstruct(cls: type, value: Any) -> Any:
    """Recursively reconstruct nested frozen dataclasses from a dict tree.

    Handles three nested-dataclass cases (2026-05-26 fix — previously only
    list-of-dataclass was reconstructed, single-instance was left as dict):
      • List[Dataclass]               : recurse each element
      • Dict[str, Dataclass]          : recurse each value
      • Dataclass (single instance)   : recurse directly
    """
    if value is None:
        return None
    if not is_dataclass(cls):
        return value
    if not isinstance(value, dict):
        return value
    kwargs: dict[str, Any] = {}
    for field_name, inner_dc, kind in _reconstruction_plan(cls):
        if field_name not in value:
            continue
        raw = value[field_name]
        if inner_dc is not None:
            if isinstance(raw, list):
                kwargs[field_name] = [
                    _reconstruct(inner_dc, item) if isinstance(item, dict) else item
                    for item in raw
                ]
            elif isinstance(raw, dict):
                # F-001 fix (2026-05-26): the previous heuristic inspected
                # `raw` contents to choose between dict-of-Dataclass vs
                # single-Dataclass. That heuristic is undefined for empty
                # dicts — it took the wrong branch and crashed iter_records
                # (silently skipping every EvidenceSummary with an empty
                # diagnostic_axis_stats, observed in BetV1 fix3: 5 records
                # on disk, 0 read back). Now we look at the field type
                # annotation, which carries the wrapper kind unambiguously.
                if kind == "dict":
                    kwargs[field_name] = {
                        k: _reconstruct(inner_dc, v) if isinstance(v, dict) else v
                        for k, v in raw.items()
                    }
                else:
                    kwargs[field_name] = _reconstruct(inner_dc, raw)
            else:
                kwargs[field_name] = raw
        else:
            kwargs[field_name] = raw
    return cls(**kwargs)


class Archive:
    """Append-only T-ReX archive.

    Single-process writer is the common case. Multi-process appends are
    safe per POSIX O_APPEND for line-sized writes (< 4 KB typically),
    but readers must defensively skip malformed lines.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._seen_result_ids: set[str] | None = None
        self._seen_result_offset = 0
        self._seen_result_file_id: tuple[int, int] | None = None
        self._read_skip_counts: dict[str, int] = {}
        self._read_cache: dict[type, tuple[tuple[int, int], tuple[Any, ...]]] = {}

    # ---- write -----------------------------------------------------------

    def append(self, record: Any) -> None:
        if not is_dataclass(record):
            raise TypeError(f"archive.append expects a frozen dataclass, got {type(record)}")
        cls = type(record)
        fname = RECORD_FILES.get(cls)
        if fname is None:
            raise ValueError(
                f"no archive file mapping for {cls.__name__}. "
                "Add to RECORD_FILES if intentional."
            )
        path = self.root / fname
        line = json.dumps(to_jsonable(record), sort_keys=True) + "\n"
        if cls is ResultRecord:
            lock_path = self.root / f"{fname}.lock"
            lock_fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT, 0o644)
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
                # Refresh only bytes appended since this Archive instance last
                # held the lock. This preserves process-safe duplicate checks
                # without an O(N^2) full JSONL rescan over a long campaign.
                self._refresh_seen_result_ids(path)
                self._assert_new_result_id(record)
                self._append_line(path, line)
                assert self._seen_result_ids is not None
                self._seen_result_ids.add(record.result_id)
                st = path.stat()
                self._seen_result_file_id = (st.st_dev, st.st_ino)
                self._seen_result_offset = st.st_size
                self._read_cache.pop(cls, None)
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                os.close(lock_fd)
            return
        self._append_line(path, line)
        self._read_cache.pop(cls, None)

    def _append_line(self, path: Path, line: str) -> None:
        # O_APPEND ensures atomic line-sized writes across processes.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, line.encode("utf-8"))
        finally:
            os.close(fd)

    def _refresh_seen_result_ids(self, path: Path) -> None:
        """Load new ResultRecord IDs since the previous lock-protected scan."""
        try:
            st = path.stat()
        except FileNotFoundError:
            self._seen_result_ids = set()
            self._seen_result_offset = 0
            self._seen_result_file_id = None
            return

        file_id = (st.st_dev, st.st_ino)
        reset = (
            self._seen_result_ids is None
            or self._seen_result_file_id != file_id
            or st.st_size < self._seen_result_offset
        )
        if reset:
            self._seen_result_ids = set()
            self._seen_result_offset = 0
        assert self._seen_result_ids is not None
        if st.st_size == self._seen_result_offset:
            self._seen_result_file_id = file_id
            return

        with path.open() as f:
            f.seek(self._seen_result_offset)
            for line in f:
                try:
                    rid = json.loads(line).get("result_id")
                except (json.JSONDecodeError, AttributeError):
                    continue
                if isinstance(rid, str):
                    self._seen_result_ids.add(rid)
            self._seen_result_offset = f.tell()
        self._seen_result_file_id = file_id

    def _assert_new_result_id(self, record: ResultRecord) -> None:
        assert self._seen_result_ids is not None
        if record.result_id in self._seen_result_ids:
            raise ValueError(
                f"duplicate ResultRecord.result_id {record.result_id!r}; "
                "result IDs must be globally unique within an archive"
            )

    def append_many(self, records: Iterable[Any]) -> None:
        for r in records:
            self.append(r)

    # ---- read ------------------------------------------------------------

    def _record_read_skip(self, cls: type, reason: str) -> None:
        fname = RECORD_FILES.get(cls, cls.__name__)
        key = f"{fname}:{reason}"
        self._read_skip_counts[key] = self._read_skip_counts.get(key, 0) + 1

    def read_skip_counts(self) -> dict[str, int]:
        """Return counts of JSON/schema lines skipped while reading archives."""
        return dict(self._read_skip_counts)

    def iter_records(self, cls: type) -> Iterator[Any]:
        fname = RECORD_FILES.get(cls)
        if fname is None:
            raise ValueError(f"no archive file mapping for {cls.__name__}")
        path = self.root / fname
        try:
            st = path.stat()
        except FileNotFoundError:
            return
        cache_key = (int(st.st_size), int(st.st_mtime_ns))
        cached = self._read_cache.get(cls)
        if cached is not None and cached[0] == cache_key:
            for rec in cached[1]:
                # Return a fresh object so live clustering/parser code can add
                # transient bins without mutating the cache snapshot.
                yield copy.deepcopy(rec)
            return

        records: list[Any] = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    # partial / corrupt line — skip rather than crash
                    self._record_read_skip(cls, "json_decode")
                    continue
                try:
                    records.append(_reconstruct(cls, d))
                except (TypeError, KeyError, ValueError):
                    self._record_read_skip(cls, "schema_drift")
                    continue
        # Cache an immutable snapshot that is isolated from the objects yielded
        # below; callers are allowed to mutate returned dataclass dict fields.
        self._read_cache[cls] = (cache_key, tuple(copy.deepcopy(records)))
        for rec in records:
            yield rec

    def count(self, cls: type) -> int:
        return sum(1 for _ in self.iter_records(cls))

    # ---- convenience -----------------------------------------------------

    def retrieve_active_hypotheses(
        self, *, target_id: str | None = None, limit: int | None = 8,
        include_terminal: bool = True,
    ) -> list[HypothesisCard]:
        """Return latest non-retired rows, active first and then newest.

        Supported/contradicted outcomes are included by default for prompt memory.
        Lifecycle maintenance passes ``include_terminal=False, limit=None`` so
        terminal memory cannot hide an older active card from TTL evaluation.
        """
        latest: dict[str, tuple[int, HypothesisCard]] = {}
        for idx, h in enumerate(self.iter_records(HypothesisCard)):
            if target_id and h.target_id != target_id:
                continue
            # Archive is append-only; later rows with the same hypothesis_id are
            # lifecycle updates and must replace earlier copies.
            latest[h.hypothesis_id] = (idx, h)
        allowed = {"active", "supported", "contradicted"} if include_terminal else {"active"}
        out = [h for _, h in latest.values() if h.status in allowed]
        # Active work must not be displaced from bounded Planner memory by a
        # burst of terminal outcomes. Within each group, keep newest first.
        out.sort(
            key=lambda h: (
                h.status == "active",
                latest[h.hypothesis_id][0],
                h.tick_created,
            ),
            reverse=True,
        )
        return out if limit is None else out[:limit]

    def latest_evidence(self, target_id: str | None = None) -> EvidenceSummary | None:
        latest = None
        for e in self.iter_records(EvidenceSummary):
            if target_id and e.target_id != target_id:
                continue
            latest = e
        return latest

    # ---- introspection ---------------------------------------------------

    def summary(self) -> dict[str, int]:
        return {RECORD_FILES[cls]: self.count(cls) for cls in RECORD_FILES}

    def __repr__(self) -> str:
        return f"Archive(root={self.root!r})"


# ---- shadow-mode helpers ----------------------------------------------------


def open_archive(root: Path | str) -> Archive:
    return Archive(root)


def new_run_archive(
    base: Path | str | None = None,
    run_label: str | None = None,
) -> Archive:
    if base is None:
        base = os.environ.get("TREX_ARCHIVE_BASE", str(Path.cwd() / "runs"))
    base = Path(base).expanduser()
    label = run_label or time.strftime("v7_run_%Y%m%dT%H%M%SZ", time.gmtime())
    return Archive(base / label)
