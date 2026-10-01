from __future__ import annotations

import bisect
import pickle
import random
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from limits.storage import MemoryStorage, memory as memory_module
from limits.storage.memory import Entry


class TestSerialization:
    def test_pickle(self):
        storage = MemoryStorage()
        assert 1 == storage.incr("test", 60)
        assert storage.acquire_entry("moving_test", 2, 60)
        dump = pickle.dumps(storage)
        restored = pickle.loads(dump)
        assert 2 == restored.incr("test", 60)
        assert restored.acquire_entry("moving_test", 2, 60)
        assert not restored.acquire_entry("moving_test", 2, 60)


class LegacyEntry:
    def __init__(self, expiry: float, atime: float) -> None:
        self.atime = atime
        self.expiry = atime + expiry


class LegacyMovingWindow:
    def __init__(self) -> None:
        self.events: dict[str, list[LegacyEntry]] = {}

    def acquire(
        self, now: float, key: str, limit: int, expiry: int, amount: int = 1
    ) -> bool:
        if amount > limit:
            return False

        events = self.events.setdefault(key, [])
        try:
            entry = events[limit - amount]
        except IndexError:
            entry = None

        if entry and entry.atime >= now - expiry:
            return False

        events[:0] = [LegacyEntry(expiry, now)] * amount
        return True

    def get(
        self, now: float, key: str, limit: int, expiry: int
    ) -> tuple[float, int]:
        events = self.events.get(key, [])
        if events:
            oldest = bisect.bisect_left(
                events, -(now - expiry), key=lambda entry: -entry.atime
            )
            return events[oldest - 1].atime, oldest
        return now, 0


class Clock:
    def __init__(self, current: float = 1000.0) -> None:
        self.current = current


def make_storage(
    monkeypatch: pytest.MonkeyPatch, current: float = 1000.0
) -> tuple[MemoryStorage, Clock]:
    storage = MemoryStorage()
    storage.timer.cancel()
    storage.timer.join()
    clock = Clock(current)
    monkeypatch.setattr(memory_module.time, "time", lambda: clock.current)
    monkeypatch.setattr(storage, "_MemoryStorage__schedule_expiry", lambda: None)
    return storage, clock


@pytest.mark.parametrize("seed", range(20))
def test_weighted_events_match_legacy_algorithm(
    monkeypatch: pytest.MonkeyPatch, seed: int
) -> None:
    storage, clock = make_storage(monkeypatch)
    reference = LegacyMovingWindow()
    rng = random.Random(seed)
    key = "moving"
    limit = 8
    expiry = 5

    for _ in range(80):
        clock.current += rng.choice([0, 0, 1, 1, 2, 3])
        amount = rng.randint(1, 5)
        expected = reference.acquire(clock.current, key, limit, expiry, amount)
        assert storage.acquire_entry(key, limit, expiry, amount) is expected
        assert storage.get_moving_window(key, limit, expiry) == reference.get(
            clock.current, key, limit, expiry
        )


def test_large_weight_uses_one_event_and_rejection_is_atomic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, clock = make_storage(monkeypatch)
    key = "moving"

    assert storage.acquire_entry(key, 2_000_000, 10, amount=1_000_000)
    events = storage.events[key]
    assert len(events) == 1
    assert events[0].weight == 1_000_000
    assert storage.get_moving_window(key, 2_000_000, 10) == (
        clock.current,
        1_000_000,
    )

    assert storage.acquire_entry(key, 2_000_000, 10, amount=1_000_000)
    assert len(events) == 1
    assert events[0].weight == 2_000_000
    assert storage.get_moving_window(key, 2_000_000, 10) == (
        clock.current,
        2_000_000,
    )

    assert not storage.acquire_entry(key, 2_000_000, 10, amount=1_000_000)
    assert len(events) == 1
    assert events[0].weight == 2_000_000

    clock.current += 10
    storage._MemoryStorage__expire_events()
    assert key not in storage.events
    assert storage.get_moving_window(key, 2_000_000, 10) == (clock.current, 0)


def test_same_time_events_merge_only_when_expiry_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, _ = make_storage(monkeypatch)
    key = "moving"

    assert storage.acquire_entry(key, 20, 10, amount=3)
    assert storage.acquire_entry(key, 20, 20, amount=4)
    assert storage.acquire_entry(key, 20, 10, amount=5)

    events = storage.events[key]
    assert [(event.expiry, event.weight) for event in events] == [
        (1020.0, 4),
        (1010.0, 8),
    ]
    assert storage.get_moving_window(key, 20, 20) == (1000.0, 12)


def test_expire_events_keeps_weighted_active_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, clock = make_storage(monkeypatch)
    key = "moving"

    assert storage.acquire_entry(key, 10, 10, amount=3)
    clock.current = 1005.0
    assert storage.acquire_entry(key, 10, 10, amount=4)

    clock.current = 1011.0
    storage._MemoryStorage__expire_events()

    events = storage.events[key]
    assert len(events) == 1
    assert events[0].atime == 1005.0
    assert events[0].weight == 4
    assert storage.get_moving_window(key, 10, 10) == (1005.0, 4)
    assert storage.acquire_entry(key, 10, 10, amount=6)
    assert not storage.acquire_entry(key, 10, 10, amount=1)


def test_concurrent_weighted_acquisitions_are_atomic() -> None:
    storage = MemoryStorage()
    start = threading.Event()

    def acquire() -> bool:
        start.wait()
        return storage.acquire_entry("moving", 1000, 60, amount=100)

    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(acquire) for _ in range(32)]
        start.set()
        accepted = sum(future.result() for future in futures)

    events = storage.events["moving"]
    assert accepted == 10
    assert storage.get_moving_window("moving", 1000, 60) == (
        events[-1].atime,
        1000,
    )
    assert len(events) <= 32
    assert sum(event.weight for event in events) == 1000


def test_concurrent_decrement_never_goes_negative() -> None:
    storage = MemoryStorage()
    storage.incr("counter", 60, amount=100)

    with ThreadPoolExecutor(max_workers=20) as executor:
        values = list(executor.map(lambda _: storage.decr("counter", 7), range(20)))

    assert min(values) == 0
    assert storage.get("counter") == 0


def test_legacy_entry_unpickle_defaults_to_unit_weight() -> None:
    entry = object.__new__(Entry)
    entry.__setstate__({"atime": 1.0, "expiry": 2.0})
    assert entry.weight == 1
