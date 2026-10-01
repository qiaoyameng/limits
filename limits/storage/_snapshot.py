"""
Versioned, JSON-serializable snapshot format for in-memory storages.

A snapshot captures the remaining quota of a storage as plain data
(no locks, timers or other live objects) so that it can be persisted
before a single process service restarts and imported afterwards.

All times are expressed relative to the moment the snapshot was taken:

- counters (fixed & sliding window state) carry ``expires_in``, the
  number of seconds until the counter expires.
- events (moving window state) carry the ``age`` of each entry at
  snapshot time together with its original lifetime ``ttl``, so that
  entries of different ages keep expiring individually.

The importing side explicitly provides the ``elapsed`` offline time
(in seconds) between export and import, which makes the format
independent of the monotonic/wall clock origin of either process.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping

from limits.errors import SnapshotError

SNAPSHOT_VERSION = 1


def _is_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise SnapshotError(f"duplicate key in snapshot: {key!r}")
        result[key] = value
    return result


def parse_snapshot(snapshot: object) -> dict[str, object]:
    """
    Parse a snapshot given as a mapping or a JSON string/bytes payload.
    """
    if isinstance(snapshot, (str, bytes, bytearray)):
        try:
            parsed = json.loads(snapshot, object_pairs_hook=_reject_duplicates)
        except json.JSONDecodeError as exc:
            raise SnapshotError(f"snapshot is not valid JSON: {exc}") from exc
    elif isinstance(snapshot, Mapping):
        parsed = dict(snapshot)
    else:
        raise SnapshotError(
            "snapshot must be a mapping or a JSON encoded string, "
            f"not {type(snapshot).__name__}"
        )
    if not isinstance(parsed, dict):
        raise SnapshotError("snapshot must be a JSON object")
    return parsed


def _validate_counter(key: object, entry: object) -> tuple[str, int, float]:
    if not isinstance(key, str):
        raise SnapshotError(f"counter key must be a string, not {key!r}")
    if not isinstance(entry, Mapping):
        raise SnapshotError(f"counter {key!r} must be a mapping")
    unknown = set(entry) - {"value", "expires_in"}
    if unknown:
        raise SnapshotError(
            f"counter {key!r} has unexpected fields: {sorted(unknown)}"
        )
    if "value" not in entry or "expires_in" not in entry:
        raise SnapshotError(
            f"counter {key!r} must define 'value' and 'expires_in'"
        )
    value = entry["value"]
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise SnapshotError(
            f"counter {key!r} has an invalid value (negative weights "
            f"are not allowed): {value!r}"
        )
    expires_in = entry["expires_in"]
    if not _is_number(expires_in) or expires_in < 0:
        raise SnapshotError(
            f"counter {key!r} has an invalid 'expires_in': {expires_in!r}"
        )
    return key, value, float(expires_in)


def _validate_event(key: object, entry: object) -> tuple[str, float, float]:
    if not isinstance(entry, Mapping):
        raise SnapshotError(f"event of {key!r} must be a mapping")
    unknown = set(entry) - {"age", "ttl"}
    if unknown:
        raise SnapshotError(
            f"event of {key!r} has unexpected fields: {sorted(unknown)}"
        )
    if "age" not in entry or "ttl" not in entry:
        raise SnapshotError(f"event of {key!r} must define 'age' and 'ttl'")
    age = entry["age"]
    if not _is_number(age) or age < 0:
        raise SnapshotError(f"event of {key!r} has an invalid 'age': {age!r}")
    ttl = entry["ttl"]
    if not _is_number(ttl) or ttl <= 0:
        raise SnapshotError(f"event of {key!r} has an invalid 'ttl': {ttl!r}")
    return key, float(age), float(ttl)


def validate_snapshot(
    data: Mapping[str, object],
) -> tuple[dict[str, tuple[int, float]], dict[str, list[tuple[float, float]]]]:
    """
    Fully validate a parsed snapshot and return its normalized content:
    a mapping of counter key to ``(value, expires_in)`` and a mapping of
    event key to a list of ``(age, ttl)`` entries.
    """
    version = data.get("version")
    if version != SNAPSHOT_VERSION:
        raise SnapshotError(
            f"unsupported snapshot version: {version!r} "
            f"(expected {SNAPSHOT_VERSION})"
        )
    unknown = set(data) - {"version", "counters", "events"}
    if unknown:
        raise SnapshotError(f"snapshot has unexpected fields: {sorted(unknown)}")
    counters_data = data.get("counters", {})
    events_data = data.get("events", {})
    if not isinstance(counters_data, Mapping):
        raise SnapshotError("'counters' must be a mapping")
    if not isinstance(events_data, Mapping):
        raise SnapshotError("'events' must be a mapping")

    counters: dict[str, tuple[int, float]] = {}
    for key, entry in counters_data.items():
        normalized_key, value, expires_in = _validate_counter(key, entry)
        counters[normalized_key] = (value, expires_in)

    events: dict[str, list[tuple[float, float]]] = {}
    for key, entries in events_data.items():
        if not isinstance(key, str):
            raise SnapshotError(f"event key must be a string, not {key!r}")
        if not isinstance(entries, list):
            raise SnapshotError(f"events of {key!r} must be a list")
        events[key] = [
            (age, ttl)
            for _, age, ttl in (_validate_event(key, entry) for entry in entries)
        ]

    return counters, events


def validate_elapsed(elapsed: object) -> float:
    if not _is_number(elapsed) or elapsed < 0:
        raise SnapshotError(
            f"elapsed time must be a finite, non-negative number: {elapsed!r}"
        )
    return float(elapsed)


def validate_namespaces(namespaces: Iterable[str] | None) -> frozenset[str] | None:
    if namespaces is None:
        return None
    if isinstance(namespaces, str):
        raise SnapshotError("namespaces must be an iterable of strings")
    validated = frozenset(namespaces)
    for namespace in validated:
        if not isinstance(namespace, str) or not namespace:
            raise SnapshotError(
                f"namespaces must be non-empty strings: {namespace!r}"
            )
    return validated


def matches_namespace(key: str, namespaces: frozenset[str] | None) -> bool:
    if namespaces is None:
        return True
    return any(
        key == namespace or key.startswith(namespace + "/")
        for namespace in namespaces
    )


def dump_snapshot(
    counters: Iterable[tuple[str, int]],
    expirations: Mapping[str, float],
    events: Iterable[tuple[str, list[tuple[float, float]]]],
    now: float,
) -> dict[str, object]:
    """
    Build a versioned snapshot from raw storage content.

    :param counters: ``(key, value)`` pairs of the fixed/sliding window counters
    :param expirations: mapping of counter key to its absolute expiry timestamp
    :param events: ``(key, [(atime, expiry), ...])`` pairs of moving window entries
    :param now: the timestamp the snapshot is taken at
    """
    dumped_counters: dict[str, object] = {}
    for key, value in counters:
        remaining = expirations.get(key, now) - now
        if value > 0 and remaining > 0:
            dumped_counters[key] = {"value": int(value), "expires_in": remaining}

    dumped_events: dict[str, object] = {}
    for key, entries in events:
        dumped_entries = []
        for atime, expiry in entries:
            age = now - atime
            ttl = expiry - atime
            if 0 <= age < ttl:
                dumped_entries.append({"age": age, "ttl": ttl})
        if dumped_entries:
            dumped_events[key] = dumped_entries

    return {
        "version": SNAPSHOT_VERSION,
        "counters": dumped_counters,
        "events": dumped_events,
    }


def load_snapshot(
    snapshot: object,
    *,
    elapsed: float,
    namespaces: Iterable[str] | None,
    now: float,
) -> tuple[dict[str, int], dict[str, float], dict[str, list[tuple[float, float]]]]:
    """
    Fully validate a snapshot and materialize it against ``now``.

    Returns ``(counters, expirations, events)`` where counters maps a key
    to its value, expirations maps a counter key to its new absolute expiry
    and events maps a key to a list of ``(atime, expiry)`` entries.

    Entries that expired while offline (``elapsed``) are dropped instead of
    being resurrected. When ``namespaces`` is provided, only entries
    belonging to those namespaces are materialized.
    """
    offline = validate_elapsed(elapsed)
    selected = validate_namespaces(namespaces)
    counters_data, events_data = validate_snapshot(parse_snapshot(snapshot))

    counters: dict[str, int] = {}
    expirations: dict[str, float] = {}
    for key, (value, expires_in) in counters_data.items():
        if not matches_namespace(key, selected):
            continue
        remaining = expires_in - offline
        if remaining > 0:
            counters[key] = value
            expirations[key] = now + remaining

    events: dict[str, list[tuple[float, float]]] = {}
    for key, entries in events_data.items():
        if not matches_namespace(key, selected):
            continue
        restored = []
        for age, ttl in entries:
            aged = age + offline
            if aged < ttl:
                atime = now - aged
                restored.append((atime, atime + ttl))
        if restored:
            events[key] = restored

    return counters, expirations, events
