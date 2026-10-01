from __future__ import annotations

import json
import pickle

import pytest

from limits.limits import RateLimitItemPerMinute
from limits.storage import MemoryStorage
from limits.errors import SnapshotError
from limits.strategies import (
    FixedWindowRateLimiter,
    MovingWindowRateLimiter,
    SlidingWindowCounterRateLimiter,
)
from tests.utils import FakeClock


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


class TestSnapshots:
    def test_roundtrip_between_clocks(self, monkeypatch):
        clock_a = FakeClock(10_000.0)
        monkeypatch.setattr("limits.storage.memory.time", clock_a)
        storage_a = MemoryStorage()

        fixed = RateLimitItemPerMinute(5, namespace="fixed")
        moving = RateLimitItemPerMinute(3, namespace="moving")
        sliding = RateLimitItemPerMinute(4, namespace="sliding")
        fixed_a = FixedWindowRateLimiter(storage_a)
        moving_a = MovingWindowRateLimiter(storage_a)
        sliding_a = SlidingWindowCounterRateLimiter(storage_a)

        assert all(fixed_a.hit(fixed) for _ in range(3))
        assert moving_a.hit(moving)
        clock_a.advance(10)
        assert moving_a.hit(moving)
        assert all(sliding_a.hit(sliding) for _ in range(2))

        # the snapshot must be plain, JSON-serializable data
        snapshot = json.loads(json.dumps(storage_a.export_snapshot()))
        assert snapshot["version"] == 1

        # 30 seconds of downtime, then a restart on a different clock
        clock_a.advance(30)
        clock_b = FakeClock(clock_a.time())
        monkeypatch.setattr("limits.storage.memory.time", clock_b)
        storage_b = MemoryStorage()
        storage_b.import_snapshot(snapshot, elapsed=30)

        fixed_b = FixedWindowRateLimiter(storage_b)
        moving_b = MovingWindowRateLimiter(storage_b)
        sliding_b = SlidingWindowCounterRateLimiter(storage_b)

        # the imported storage behaves like the original one that kept running
        for limiter_a, limiter_b, item in [
            (fixed_a, fixed_b, fixed),
            (moving_a, moving_b, moving),
            (sliding_a, sliding_b, sliding),
        ]:
            stats_a = limiter_a.get_window_stats(item)
            stats_b = limiter_b.get_window_stats(item)
            assert stats_b.remaining == stats_a.remaining
            assert stats_b.reset_time == pytest.approx(stats_a.reset_time)

        # fixed window: 2 of 5 remaining, then rejections
        assert fixed_b.hit(fixed) == fixed_a.hit(fixed)
        assert fixed_b.hit(fixed) == fixed_a.hit(fixed)
        assert not fixed_b.hit(fixed)
        assert not fixed_a.hit(fixed)
        assert fixed_b.get_window_stats(fixed).remaining == 0

        # moving window: 1 of 3 remaining, then rejections
        assert moving_b.hit(moving) == moving_a.hit(moving)
        assert not moving_b.hit(moving)
        assert not moving_a.hit(moving)

        # sliding window: weighted remainder of the previous window is kept
        assert sliding_b.hit(sliding) == sliding_a.hit(sliding)
        assert sliding_b.get_window_stats(sliding).remaining == (
            sliding_a.get_window_stats(sliding).remaining
        )

        # let the first fixed window and the oldest moving entry expire
        clock_b.advance(25)

        # fixed window: a fresh window starts
        assert fixed_b.get_window_stats(fixed).remaining == 5
        assert fixed_b.hit(fixed) == fixed_a.hit(fixed)

        # moving window: only the oldest entry expired, the younger
        # ones still count individually
        stats_b = moving_b.get_window_stats(moving)
        assert stats_b.remaining == moving_a.get_window_stats(moving).remaining == 1
        assert moving_b.hit(moving) == moving_a.hit(moving)
        assert not moving_b.hit(moving)
        assert not moving_a.hit(moving)

        # sliding window: previous window weight decays identically
        stats_b = sliding_b.get_window_stats(sliding)
        stats_a = sliding_a.get_window_stats(sliding)
        assert stats_b.remaining == stats_a.remaining
        assert stats_b.reset_time == pytest.approx(stats_a.reset_time)

    def test_roundtrip_with_different_clock_origin(self, monkeypatch):
        clock_a = FakeClock(10_000.0)
        monkeypatch.setattr("limits.storage.memory.time", clock_a)
        storage_a = MemoryStorage()

        fixed = RateLimitItemPerMinute(3, namespace="fixed")
        moving = RateLimitItemPerMinute(2, namespace="moving")
        fixed_a = FixedWindowRateLimiter(storage_a)
        moving_a = MovingWindowRateLimiter(storage_a)

        assert all(fixed_a.hit(fixed) for _ in range(2))
        assert moving_a.hit(moving)
        clock_a.advance(5)
        assert moving_a.hit(moving)

        snapshot = storage_a.export_snapshot()

        # the importing process lives on a completely different clock origin
        clock_b = FakeClock(9_000_000.0)
        monkeypatch.setattr("limits.storage.memory.time", clock_b)
        storage_b = MemoryStorage()
        storage_b.import_snapshot(snapshot, elapsed=15)

        fixed_b = FixedWindowRateLimiter(storage_b)
        moving_b = MovingWindowRateLimiter(storage_b)

        # fixed window: 1 of 3 remaining, reset relative to the new clock
        stats = fixed_b.get_window_stats(fixed)
        assert stats.remaining == 1
        assert stats.reset_time == pytest.approx(clock_b.time() + 45)
        assert fixed_b.hit(fixed)
        assert not fixed_b.hit(fixed)

        # moving window: both entries (aged 20 and 15) still count
        assert not moving_b.hit(moving)
        # after 41 more seconds only the oldest entry expired
        clock_b.advance(41)
        assert moving_b.hit(moving)
        assert not moving_b.hit(moving)
        stats = moving_b.get_window_stats(moving)
        assert stats.remaining == 0
        assert stats.reset_time == pytest.approx(clock_b.time() + 4)

    def test_expired_entries_are_not_resurrected(self, monkeypatch):
        clock_a = FakeClock(1_000.0)
        monkeypatch.setattr("limits.storage.memory.time", clock_a)
        storage_a = MemoryStorage()

        fixed = RateLimitItemPerMinute(1, namespace="fixed")
        moving = RateLimitItemPerMinute(1, namespace="moving")
        assert FixedWindowRateLimiter(storage_a).hit(fixed)
        assert MovingWindowRateLimiter(storage_a).hit(moving)

        snapshot = storage_a.export_snapshot()

        clock_b = FakeClock(5_000.0)
        monkeypatch.setattr("limits.storage.memory.time", clock_b)
        storage_b = MemoryStorage()
        storage_b.import_snapshot(snapshot, elapsed=61)

        # everything expired during the downtime: the full quota is back
        assert storage_b.export_snapshot() == {
            "version": 1,
            "counters": {},
            "events": {},
        }
        fixed_b = FixedWindowRateLimiter(storage_b)
        moving_b = MovingWindowRateLimiter(storage_b)
        assert fixed_b.get_window_stats(fixed).remaining == 1
        assert moving_b.get_window_stats(moving).remaining == 1
        assert fixed_b.hit(fixed)
        assert moving_b.hit(moving)

    def test_import_replaces_only_given_namespaces(self, monkeypatch):
        clock = FakeClock(100.0)
        monkeypatch.setattr("limits.storage.memory.time", clock)
        storage_a = MemoryStorage()
        one = RateLimitItemPerMinute(5, namespace="one")
        two = RateLimitItemPerMinute(5, namespace="two")
        moving_one = RateLimitItemPerMinute(5, namespace="one")

        limiter_a = FixedWindowRateLimiter(storage_a)
        assert all(limiter_a.hit(one) for _ in range(2))
        assert all(limiter_a.hit(two) for _ in range(3))
        assert MovingWindowRateLimiter(storage_a).hit(moving_one)
        snapshot = storage_a.export_snapshot()

        storage_b = MemoryStorage()
        limiter_b = FixedWindowRateLimiter(storage_b)
        three = RateLimitItemPerMinute(5, namespace="three")
        assert all(limiter_b.hit(one) for _ in range(4))
        assert limiter_b.hit(two)
        assert all(limiter_b.hit(three) for _ in range(2))

        storage_b.import_snapshot(snapshot, namespaces=["one"])

        # namespace "one" was replaced by the snapshot content
        assert storage_b.get(one.key_for()) == 2
        assert storage_b.get_moving_window(moving_one.key_for(), 5, 60)[1] == 1
        # namespace "two" was left untouched and not imported
        assert storage_b.get(two.key_for()) == 1
        # unrelated namespaces were left untouched
        assert storage_b.get(three.key_for()) == 2

        # importing again without namespaces replaces everything
        storage_b.import_snapshot(snapshot)
        assert storage_b.get(one.key_for()) == 2
        assert storage_b.get(two.key_for()) == 3
        assert storage_b.get(three.key_for()) == 0

    @pytest.mark.parametrize(
        "snapshot",
        [
            {"version": 2, "counters": {}, "events": {}},
            {"counters": {}, "events": {}},
            {"version": 1, "counters": {}, "events": {}, "extra": 1},
            {"version": 1, "counters": {"k": {"value": -1, "expires_in": 1}}},
            {"version": 1, "counters": {"k": {"value": 1.5, "expires_in": 1}}},
            {"version": 1, "counters": {"k": {"value": True, "expires_in": 1}}},
            {"version": 1, "counters": {"k": {"value": 1, "expires_in": -1}}},
            {"version": 1, "counters": {"k": {"value": 1}}},
            {"version": 1, "counters": {"k": 1}},
            {"version": 1, "counters": []},
            {"version": 1, "events": {"k": [{"age": -1, "ttl": 5}]}},
            {"version": 1, "events": {"k": [{"age": 0, "ttl": 0}]}},
            {"version": 1, "events": {"k": [{"age": 0}]}},
            {"version": 1, "events": {"k": {"age": 0, "ttl": 1}}},
            '{"version": 1, "counters": {"k": {"value": 1, "expires_in": 1}, '
            '"k": {"value": 2, "expires_in": 1}}',
            "not a snapshot",
            "[1, 2, 3]",
            [1, 2, 3],
            42,
        ],
    )
    def test_invalid_snapshots_are_rejected(self, monkeypatch, snapshot):
        clock = FakeClock(100.0)
        monkeypatch.setattr("limits.storage.memory.time", clock)
        storage = MemoryStorage()
        item = RateLimitItemPerMinute(5, namespace="kept")
        assert FixedWindowRateLimiter(storage).hit(item)
        before = storage.export_snapshot()

        with pytest.raises(SnapshotError):
            storage.import_snapshot(snapshot)

        # a broken snapshot must not corrupt the existing state
        assert storage.export_snapshot() == before
        assert storage.get(item.key_for()) == 1

    def test_invalid_elapsed_and_namespaces_are_rejected(self, monkeypatch):
        clock = FakeClock(100.0)
        monkeypatch.setattr("limits.storage.memory.time", clock)
        storage = MemoryStorage()
        storage.incr("kept", 60)
        before = storage.export_snapshot()

        with pytest.raises(SnapshotError):
            storage.import_snapshot(before, elapsed=-1)
        with pytest.raises(SnapshotError):
            storage.import_snapshot(before, namespaces=[""])
        with pytest.raises(SnapshotError):
            storage.import_snapshot(before, namespaces="one")
        with pytest.raises(SnapshotError):
            storage.import_snapshot(before, namespaces=["ok", 1])

        assert storage.export_snapshot() == before
