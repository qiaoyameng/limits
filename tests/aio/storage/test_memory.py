from __future__ import annotations

import json
import pickle

import pytest

from limits.aio.storage import MemoryStorage
from limits.limits import RateLimitItemPerMinute
from limits.storage import MemoryStorage as SyncMemoryStorage
from limits.errors import SnapshotError
from limits.strategies import FixedWindowRateLimiter, MovingWindowRateLimiter
from tests.utils import FakeClock


class TestSerialization:
    async def test_pickle(self):
        storage = MemoryStorage()
        assert 1 == await storage.incr("test", 60)
        assert await storage.acquire_entry("moving_test", 2, 60)
        dump = pickle.dumps(storage)
        restored = pickle.loads(dump)
        assert 2 == await restored.incr("test", 60)
        assert await restored.acquire_entry("moving_test", 2, 60)
        assert not await restored.acquire_entry("moving_test", 2, 60)


class TestSnapshots:
    async def test_roundtrip_between_clocks(self, monkeypatch):
        clock_a = FakeClock(10_000.0)
        monkeypatch.setattr("limits.aio.storage.memory.time", clock_a)
        storage_a = MemoryStorage()

        fixed = RateLimitItemPerMinute(5, namespace="fixed")
        moving = RateLimitItemPerMinute(2, namespace="moving")
        assert 3 == await storage_a.incr(fixed.key_for(), 60, amount=3)
        assert await storage_a.acquire_entry(moving.key_for(), 2, 60)
        clock_a.advance(10)

        # the snapshot must be plain, JSON-serializable data
        snapshot = json.loads(json.dumps(await storage_a.export_snapshot()))
        assert snapshot["version"] == 1

        # restart on a clock with a completely different origin
        clock_b = FakeClock(900_000.0)
        monkeypatch.setattr("limits.aio.storage.memory.time", clock_b)
        storage_b = MemoryStorage()
        await storage_b.import_snapshot(snapshot, elapsed=20)

        # fixed window counter: 3 of 5 consumed, 30 seconds remaining
        assert 3 == await storage_b.get(fixed.key_for())
        assert await storage_b.get_expiry(fixed.key_for()) == pytest.approx(
            clock_b.time() + 30
        )

        # moving window entry aged by the offline time
        start, count = await storage_b.get_moving_window(moving.key_for(), 2, 60)
        assert count == 1
        assert start == pytest.approx(clock_b.time() - 30)
        assert await storage_b.acquire_entry(moving.key_for(), 2, 60)
        assert not await storage_b.acquire_entry(moving.key_for(), 2, 60)

        # the imported entry expires individually
        clock_b.advance(31)
        _, count = await storage_b.get_moving_window(moving.key_for(), 2, 60)
        assert count == 1

    async def test_invalid_snapshots_are_rejected(self, monkeypatch):
        clock = FakeClock(100.0)
        monkeypatch.setattr("limits.aio.storage.memory.time", clock)
        storage = MemoryStorage()
        assert 1 == await storage.incr("kept", 60)

        with pytest.raises(SnapshotError):
            await storage.import_snapshot({"version": 2})
        with pytest.raises(SnapshotError):
            await storage.import_snapshot(
                '{"version": 1, "counters": '
                '{"a": {"value": -1, "expires_in": 1}}}'
            )
        with pytest.raises(SnapshotError):
            await storage.import_snapshot(
                '{"version": 1, "counters": '
                '{"a": {"value": 1, "expires_in": 1}, '
                '"a": {"value": 2, "expires_in": 1}}}'
            )

        # a broken snapshot must not corrupt the existing state
        assert 1 == await storage.get("kept")

    async def test_sync_export_async_import(self, monkeypatch):
        clock_sync = FakeClock(10_000.0)
        monkeypatch.setattr("limits.storage.memory.time", clock_sync)
        sync_storage = SyncMemoryStorage()
        item = RateLimitItemPerMinute(4, namespace="shared")
        fixed_sync = FixedWindowRateLimiter(sync_storage)
        moving_sync = MovingWindowRateLimiter(sync_storage)
        assert all(fixed_sync.hit(item) for _ in range(3))
        assert moving_sync.hit(item)
        snapshot = json.dumps(sync_storage.export_snapshot())

        clock_async = FakeClock(50_000.0)
        monkeypatch.setattr("limits.aio.storage.memory.time", clock_async)
        async_storage = MemoryStorage()
        await async_storage.import_snapshot(snapshot, elapsed=10)

        assert 3 == await async_storage.get(item.key_for())
        assert await async_storage.get_expiry(item.key_for()) == pytest.approx(
            clock_async.time() + 50
        )
        _, count = await async_storage.get_moving_window(item.key_for(), 4, 60)
        assert count == 1

        # and back: async export -> sync import
        snapshot_back = json.loads(json.dumps(await async_storage.export_snapshot()))
        monkeypatch.setattr("limits.storage.memory.time", FakeClock(70_000.0))
        restored = SyncMemoryStorage()
        restored.import_snapshot(snapshot_back, elapsed=5)
        assert 3 == restored.get(item.key_for())
        assert restored.get_expiry(item.key_for()) == pytest.approx(70_000 + 45)
