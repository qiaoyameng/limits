from __future__ import annotations

import asyncio
import contextlib
import json
import time
from unittest import mock

import pytest

import limits.aio.storage.memory as aio_memory
import limits.storage.memory as sync_memory
from limits.aio.storage import MemoryStorage as AsyncMemoryStorage
from limits.errors import SnapshotError
from limits.limits import RateLimitItemPerMinute
from limits.storage import MemoryStorage
from limits.strategies import (
    FixedWindowRateLimiter,
    MovingWindowRateLimiter,
    SlidingWindowCounterRateLimiter,
)


class FakeClock:
    """
    Stand-in for the ``time`` module of a storage, with an arbitrary
    clock origin, so that time semantics can be tested without waiting
    for real time (or a real process restart) to pass.
    """

    def __init__(self, now: float) -> None:
        self.now = now

    def time(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@contextlib.contextmanager
def use_clock(module, clock):
    with mock.patch.object(module, "time", clock):
        yield clock


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()


def future_clock(offset: float) -> FakeClock:
    # keep fake clocks in the future of the real clock so that the
    # storage's real-time background expiry never interferes
    return FakeClock(time.time() + offset)


class TestSnapshotFormat:
    def test_snapshot_is_json_serializable(self):
        clock = future_clock(100_000)
        storage = MemoryStorage()
        with use_clock(sync_memory, clock):
            storage.incr("LIMITER/fixed/10/1/minute", 60)
            storage.acquire_entry("LIMITER/moving", 2, 60)
            snapshot = storage.export_snapshot()

        assert snapshot["version"] == 1
        transferred = json.loads(json.dumps(snapshot))

        target = MemoryStorage()
        with use_clock(sync_memory, future_clock(200_000)):
            target.import_snapshot(transferred, elapsed=0)
            assert target.get("LIMITER/fixed/10/1/minute") == 1
            assert target.get_moving_window("LIMITER/moving", 2, 60)[1] == 1

    def test_export_prunes_expired_entries(self):
        clock = future_clock(100_000)
        storage = MemoryStorage()
        with use_clock(sync_memory, clock):
            storage.incr("LIMITER/fixed/10/1/minute", 30)
            storage.acquire_entry("LIMITER/moving", 2, 30)
            clock.advance(40)
            snapshot = storage.export_snapshot()

        assert snapshot["counters"] == []
        assert snapshot["events"] == []


class TestFixedWindowSnapshot:
    def test_roundtrip_between_different_clocks(self):
        clock_a = future_clock(100_000)
        clock_b = future_clock(900_000)
        source, target = MemoryStorage(), MemoryStorage()
        limit = RateLimitItemPerMinute(5)

        with use_clock(sync_memory, clock_a):
            limiter_a = FixedWindowRateLimiter(source)
            assert all(limiter_a.hit(limit, "user") for _ in range(3))
            snapshot = source.export_snapshot()

        with use_clock(sync_memory, clock_b):
            target.import_snapshot(snapshot, elapsed=10)
            limiter_b = FixedWindowRateLimiter(target)

            stats = limiter_b.get_window_stats(limit, "user")
            assert stats.remaining == 2
            # 10 offline seconds were fast-forwarded on import
            assert stats.reset_time == pytest.approx(clock_b.now + 50)

            assert limiter_b.hit(limit, "user")
            assert limiter_b.hit(limit, "user")
            assert not limiter_b.hit(limit, "user")

            clock_b.advance(51)
            stats = limiter_b.get_window_stats(limit, "user")
            assert stats.remaining == 5
            assert limiter_b.hit(limit, "user")

    def test_roundtrip_through_a_second_export(self):
        clock_a = future_clock(100_000)
        clock_b = future_clock(500_000)
        clock_c = future_clock(900_000)
        limit = RateLimitItemPerMinute(5)
        storage_a, storage_b, storage_c = (
            MemoryStorage(),
            MemoryStorage(),
            MemoryStorage(),
        )

        with use_clock(sync_memory, clock_a):
            limiter_a = FixedWindowRateLimiter(storage_a)
            assert all(limiter_a.hit(limit, "user") for _ in range(3))
            snapshot_a = storage_a.export_snapshot()

        with use_clock(sync_memory, clock_b):
            storage_b.import_snapshot(snapshot_a, elapsed=10)
            clock_b.advance(5)
            snapshot_b = storage_b.export_snapshot()

        with use_clock(sync_memory, clock_c):
            storage_c.import_snapshot(snapshot_b, elapsed=2)
            limiter_c = FixedWindowRateLimiter(storage_c)
            stats = limiter_c.get_window_stats(limit, "user")
            assert stats.remaining == 2
            # 60 - (10 + 5 + 2) seconds left on the original window
            assert stats.reset_time == pytest.approx(clock_c.now + 43)


class TestMovingWindowSnapshot:
    def test_entries_of_different_ages_expire_separately(self):
        clock_a = future_clock(100_000)
        clock_b = future_clock(900_000)
        source, target = MemoryStorage(), MemoryStorage()
        limit = RateLimitItemPerMinute(3)

        with use_clock(sync_memory, clock_a):
            limiter_a = MovingWindowRateLimiter(source)
            assert limiter_a.hit(limit, "user")
            clock_a.advance(20)
            assert limiter_a.hit(limit, "user")
            clock_a.advance(20)
            assert limiter_a.hit(limit, "user")
            assert not limiter_a.hit(limit, "user")
            snapshot = source.export_snapshot()

        with use_clock(sync_memory, clock_b):
            target.import_snapshot(snapshot, elapsed=5)
            limiter_b = MovingWindowRateLimiter(target)

            # ages on the importer's clock: 45, 25 and 5 seconds
            stats = limiter_b.get_window_stats(limit, "user")
            assert stats.remaining == 0
            assert stats.reset_time == pytest.approx(clock_b.now + 15)
            assert not limiter_b.hit(limit, "user")

            clock_b.advance(16)
            stats = limiter_b.get_window_stats(limit, "user")
            assert stats.remaining == 1
            assert limiter_b.hit(limit, "user")
            assert not limiter_b.hit(limit, "user")

            clock_b.advance(20)
            stats = limiter_b.get_window_stats(limit, "user")
            assert stats.remaining == 1

            clock_b.advance(25)
            stats = limiter_b.get_window_stats(limit, "user")
            assert stats.remaining == 2

            clock_b.advance(25)
            stats = limiter_b.get_window_stats(limit, "user")
            assert stats.remaining == 3
            assert limiter_b.hit(limit, "user")


class TestSlidingWindowSnapshot:
    def test_roundtrip_between_different_clocks(self):
        expiry = 60
        # the sliding window counter derives its keys from absolute
        # window boundaries, so the two clocks are placed in the same
        # window while still having different origins
        window_start = (int(time.time()) // expiry + 100) * expiry
        clock_a = FakeClock(window_start + 10)
        clock_b = FakeClock(window_start + 20)
        source, target = MemoryStorage(), MemoryStorage()
        limit = RateLimitItemPerMinute(4)

        with use_clock(sync_memory, clock_a):
            limiter_a = SlidingWindowCounterRateLimiter(source)
            clock_a.advance(-40)
            assert limiter_a.hit(limit, "user")
            assert limiter_a.hit(limit, "user")
            clock_a.advance(40)
            assert limiter_a.hit(limit, "user")
            assert limiter_a.hit(limit, "user")
            snapshot = source.export_snapshot()

        with use_clock(sync_memory, clock_b):
            target.import_snapshot(snapshot, elapsed=5)
            limiter_b = SlidingWindowCounterRateLimiter(target)

            # weighted count: 2 * 40 / 60 (previous window) + 2 (current)
            stats = limiter_b.get_window_stats(limit, "user")
            assert stats.remaining == 1
            assert limiter_b.hit(limit, "user")
            assert not limiter_b.hit(limit, "user")

            # the previous window of the source became the current one
            clock_b.advance(45)
            stats = limiter_b.get_window_stats(limit, "user")
            assert stats.remaining == 2
            assert limiter_b.hit(limit, "user")
            assert limiter_b.hit(limit, "user")
            assert not limiter_b.hit(limit, "user")


class TestExpiredEntries:
    def test_entries_expired_while_offline_are_not_revived(self):
        clock_a = future_clock(100_000)
        clock_b = future_clock(900_000)
        source, target = MemoryStorage(), MemoryStorage()

        with use_clock(sync_memory, clock_a):
            source.incr("LIMITER/fixed/10/1/minute", 30)
            source.acquire_entry("LIMITER/moving", 2, 30)
            snapshot = source.export_snapshot()

        with use_clock(sync_memory, clock_b):
            target.import_snapshot(snapshot, elapsed=45)
            assert target.get("LIMITER/fixed/10/1/minute") == 0
            assert target.get_moving_window("LIMITER/moving", 2, 30)[1] == 0
            assert "LIMITER/fixed/10/1/minute" not in target.expirations
            assert "LIMITER/moving" not in target.events


class TestNamespaceImport:
    def test_import_replaces_only_the_given_namespace(self):
        clock_a = future_clock(100_000)
        clock_b = future_clock(900_000)
        source, target = MemoryStorage(), MemoryStorage()
        limiter_ns = RateLimitItemPerMinute(10, namespace="LIMITER")
        other_ns = RateLimitItemPerMinute(10, namespace="OTHER")

        with use_clock(sync_memory, clock_a):
            limiter_a = FixedWindowRateLimiter(source)
            assert all(limiter_a.hit(limiter_ns, "a") for _ in range(3))
            assert limiter_a.hit(other_ns, "b")
            snapshot = source.export_snapshot()

        with use_clock(sync_memory, clock_b):
            limiter_b = FixedWindowRateLimiter(target)
            assert limiter_b.hit(limiter_ns, "z")
            assert limiter_b.hit(other_ns, "b")
            assert limiter_b.hit(other_ns, "b")

            target.import_snapshot(snapshot, elapsed=0, namespace="LIMITER")

            # the LIMITER namespace was replaced by the snapshot
            assert target.get(limiter_ns.key_for("a")) == 3
            assert target.get(limiter_ns.key_for("z")) == 0
            # records of other namespaces were not imported and the
            # existing keys of other namespaces were left untouched
            assert target.get(other_ns.key_for("b")) == 2


class TestSnapshotValidation:
    @pytest.fixture
    def target_with_state(self):
        storage = MemoryStorage()
        storage.incr("LIMITER/existing/10/1/minute", 60)
        return storage

    def test_unknown_version(self, target_with_state):
        for version in (2, "1", None):
            snapshot = {"version": version, "counters": [], "events": []}
            with pytest.raises(SnapshotError):
                target_with_state.import_snapshot(snapshot, elapsed=0)
        assert target_with_state.get("LIMITER/existing/10/1/minute") == 1

    def test_negative_weight(self, target_with_state):
        snapshot = {
            "version": 1,
            "counters": [{"key": "LIMITER/a", "value": -1, "expires_in": 10}],
            "events": [],
        }
        with pytest.raises(SnapshotError):
            target_with_state.import_snapshot(snapshot, elapsed=0)
        assert target_with_state.get("LIMITER/existing/10/1/minute") == 1

    def test_negative_durations(self, target_with_state):
        snapshot = {
            "version": 1,
            "counters": [{"key": "LIMITER/a", "value": 1, "expires_in": -10}],
            "events": [
                {
                    "key": "LIMITER/b",
                    "entries": [{"age": -1, "expires_in": 10}],
                }
            ],
        }
        with pytest.raises(SnapshotError):
            target_with_state.import_snapshot(snapshot, elapsed=0)

    def test_duplicate_keys(self, target_with_state):
        record = {"key": "LIMITER/a", "value": 1, "expires_in": 10}
        snapshot = {"version": 1, "counters": [record, dict(record)], "events": []}
        with pytest.raises(SnapshotError):
            target_with_state.import_snapshot(snapshot, elapsed=0)

        event = {"key": "LIMITER/b", "entries": [{"age": 0, "expires_in": 10}]}
        snapshot = {"version": 1, "counters": [], "events": [event, dict(event)]}
        with pytest.raises(SnapshotError):
            target_with_state.import_snapshot(snapshot, elapsed=0)
        assert target_with_state.get("LIMITER/existing/10/1/minute") == 1

    def test_negative_elapsed(self, target_with_state):
        snapshot = {"version": 1, "counters": [], "events": []}
        with pytest.raises(SnapshotError):
            target_with_state.import_snapshot(snapshot, elapsed=-1)

    def test_corrupt_snapshot_is_not_partially_applied(self, target_with_state):
        snapshot = {
            "version": 1,
            "counters": [
                {"key": "LIMITER/valid", "value": 1, "expires_in": 10},
                {"key": "LIMITER/broken", "value": -1, "expires_in": 10},
            ],
            "events": [],
        }
        with pytest.raises(SnapshotError):
            target_with_state.import_snapshot(snapshot, elapsed=0)
        # the valid record was not applied either
        assert target_with_state.get("LIMITER/valid") == 0
        assert target_with_state.get("LIMITER/existing/10/1/minute") == 1


class TestSyncAsyncInterop:
    def test_sync_export_async_import(self):
        clock_a = future_clock(100_000)
        clock_b = future_clock(900_000)
        source = MemoryStorage()
        target = AsyncMemoryStorage()
        limit = RateLimitItemPerMinute(2)

        with use_clock(sync_memory, clock_a):
            limiter_a = MovingWindowRateLimiter(source)
            assert limiter_a.hit(limit, "user")
            assert source.incr("LIMITER/fixed/10/1/minute", 60) == 1
            snapshot = source.export_snapshot()

        async def import_and_check():
            with use_clock(aio_memory, clock_b):
                await target.import_snapshot(snapshot, elapsed=10)

                assert await target.get("LIMITER/fixed/10/1/minute") == 1
                start, count = await target.get_moving_window(
                    limit.key_for("user"), 2, 60
                )
                assert count == 1
                assert start == pytest.approx(clock_b.now - 10)

                assert await target.acquire_entry(limit.key_for("user"), 2, 60)
                assert not await target.acquire_entry(limit.key_for("user"), 2, 60)

        run_async(import_and_check())

    def test_async_export_sync_import(self):
        clock_a = future_clock(100_000)
        clock_b = future_clock(900_000)
        source = AsyncMemoryStorage()
        target = MemoryStorage()

        async def export():
            with use_clock(aio_memory, clock_a):
                assert await source.incr("LIMITER/fixed/10/1/minute", 60) == 1
                assert await source.acquire_entry("LIMITER/moving", 2, 60)
                return await source.export_snapshot()

        snapshot = run_async(export())
        assert json.loads(json.dumps(snapshot)) == snapshot

        with use_clock(sync_memory, clock_b):
            target.import_snapshot(snapshot, elapsed=10)
            assert target.get("LIMITER/fixed/10/1/minute") == 1
            assert target.get_moving_window("LIMITER/moving", 2, 60)[1] == 1
