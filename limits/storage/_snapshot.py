"""
Helpers to export and import the state of in-memory storages.

A snapshot is a plain, JSON-serializable mapping with an explicit
version so that it can be inspected, validated and transferred between
processes. It never contains locks, timers or any other live objects.

All timestamps are expressed relative to the moment the snapshot was
taken (remaining time to live for expirations, age for acquired
entries), which makes a snapshot independent of the clock origin of the
exporting and importing processes. The importing side explicitly
provides the time elapsed while the snapshot was offline so that
entries keep expiring on their original schedule.
"""

from __future__ import annotations

import math

import limits.typing
from limits.errors import SnapshotError

SNAPSHOT_VERSION = 1


def _is_number(value: limits.typing.Any) -> bool:  # type: ignore[explicit-any]
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _check_duration(
    value: limits.typing.Any,  # type: ignore[explicit-any]
    key: str,
    field: str,
) -> None:
    if not _is_number(value) or value < 0:
        raise SnapshotError(
            f"invalid {field} for key {key!r}: expected a non-negative number,"
            f" got {value!r}"
        )


def _check_key(
    record: limits.typing.Any,  # type: ignore[explicit-any]
    seen: set[str],
) -> str:
    if not isinstance(record, dict):
        raise SnapshotError(f"snapshot records must be mappings, got {record!r}")
    key = record.get("key")
    if not isinstance(key, str) or not key:
        raise SnapshotError(f"snapshot record has an invalid key: {key!r}")
    if key in seen:
        raise SnapshotError(f"duplicate key {key!r} in snapshot")
    seen.add(key)
    return key


def export_snapshot(
    now: float,
    storage: limits.typing.Counter[str],
    expirations: dict[str, float],
    events: dict[str, list[limits.typing.Any]],  # type: ignore[explicit-any]
) -> dict[str, limits.typing.Any]:  # type: ignore[explicit-any]
    """
    Build a versioned snapshot of the raw in-memory state.

    Entries that have already expired at ``now`` are pruned instead of
    being exported.
    """
    counters = []
    for key, value in storage.items():
        expires_in = expirations.get(key, now) - now
        if value > 0 and expires_in > 0:
            counters.append(
                {"key": key, "value": int(value), "expires_in": expires_in}
            )
    moving = []
    for key, entries in events.items():
        live = [
            {"age": now - entry.atime, "expires_in": entry.expiry - now}
            for entry in entries
            if entry.expiry > now
        ]
        if live:
            moving.append({"key": key, "entries": live})
    return {"version": SNAPSHOT_VERSION, "counters": counters, "events": moving}


def validate_snapshot(snapshot: limits.typing.Any) -> None:  # type: ignore[explicit-any]
    """
    Validate a snapshot document, raising :exc:`limits.errors.SnapshotError`
    on the first problem found.
    """
    if not isinstance(snapshot, dict):
        raise SnapshotError(f"snapshot must be a mapping, got {snapshot!r}")
    version = snapshot.get("version")
    if version != SNAPSHOT_VERSION:
        raise SnapshotError(f"unsupported snapshot version: {version!r}")
    counters = snapshot.get("counters", [])
    events = snapshot.get("events", [])
    if not isinstance(counters, list):
        raise SnapshotError("snapshot counters must be a list of records")
    if not isinstance(events, list):
        raise SnapshotError("snapshot events must be a list of records")
    seen_counters: set[str] = set()
    for record in counters:
        key = _check_key(record, seen_counters)
        value = record.get("value")
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise SnapshotError(
                f"invalid counter value for key {key!r}: expected a"
                f" non-negative integer, got {value!r}"
            )
        _check_duration(record.get("expires_in"), key, "expiry")
    seen_events: set[str] = set()
    for record in events:
        key = _check_key(record, seen_events)
        entries = record.get("entries")
        if not isinstance(entries, list):
            raise SnapshotError(
                f"invalid entries for key {key!r}: expected a list,"
                f" got {entries!r}"
            )
        for entry in entries:
            if not isinstance(entry, dict):
                raise SnapshotError(
                    f"invalid entry for key {key!r}: expected a mapping,"
                    f" got {entry!r}"
                )
            _check_duration(entry.get("age"), key, "age")
            _check_duration(entry.get("expires_in"), key, "expiry")


def plan_import(
    snapshot: limits.typing.Any,  # type: ignore[explicit-any]
    now: float,
    elapsed: float,
    namespace: str | None = None,
) -> tuple[
    dict[str, tuple[int, float]],
    dict[str, list[tuple[float, float]]],
]:
    """
    Fully validate ``snapshot`` and translate it into absolute timestamps
    on the importer's clock without touching any live state.

    :param snapshot: the snapshot document produced by :func:`export_snapshot`
    :param now: the current time on the importer's clock
    :param elapsed: seconds elapsed between the export and this import
    :param namespace: when provided, only records whose key belongs to this
     namespace (the first ``/`` separated segment) are planned

    :return: a tuple of ``{key: (value, expiry)}`` for counters and
     ``{key: [(atime, expiry), ...]}`` for moving window entries, with
     entries that expired while the snapshot was offline dropped.
    """
    validate_snapshot(snapshot)
    if not _is_number(elapsed) or elapsed < 0:
        raise SnapshotError(
            f"elapsed must be a non-negative number, got {elapsed!r}"
        )
    counters: dict[str, tuple[int, float]] = {}
    for record in snapshot.get("counters", []):
        key = record["key"]
        if namespace is not None and key.split("/", 1)[0] != namespace:
            continue
        remaining = record["expires_in"] - elapsed
        if remaining <= 0:
            continue
        counters[key] = (record["value"], now + remaining)
    events: dict[str, list[tuple[float, float]]] = {}
    for record in snapshot.get("events", []):
        key = record["key"]
        if namespace is not None and key.split("/", 1)[0] != namespace:
            continue
        entries = sorted(
            (
                (now - (entry["age"] + elapsed), now + (entry["expires_in"] - elapsed))
                for entry in record["entries"]
                if entry["expires_in"] - elapsed > 0
            ),
            reverse=True,
        )
        if entries:
            events[key] = entries
    return counters, events


def namespace_of(key: str) -> str:
    """
    Return the namespace (first ``/`` separated segment) of a rate limit key.
    """
    return key.split("/", 1)[0]
