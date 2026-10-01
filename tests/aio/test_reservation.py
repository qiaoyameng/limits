from __future__ import annotations

import asyncio
import time

import pytest

from limits.aio.storage import MemoryStorage
from limits.aio.strategies import (
    FixedWindowRateLimiter,
    MovingWindowRateLimiter,
    SlidingWindowCounterRateLimiter,
)
from limits.errors import LeaseConflictError, LeaseExpiredError
from limits.limits import RateLimitItemPerMinute, RateLimitItemPerSecond


class FakeClock:
    def __init__(self, start: float = 1_000_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def fake_clock(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(time, "time", clock)
    return clock


async def wait_until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.005)
    return predicate()


@pytest.mark.memory
class TestFixedWindowReservation:
    async def test_reserve_commit_settles_actual_cost(self, fake_clock):
        storage = MemoryStorage()
        limiter = FixedWindowRateLimiter(storage)
        item = RateLimitItemPerMinute(10)

        lease = await limiter.reserve(item, cost=8)
        assert lease is not None
        assert (await limiter.get_window_stats(item)).remaining == 2

        await limiter.commit(lease, 3)
        assert (await limiter.get_window_stats(item)).remaining == 7
        assert storage.leases[lease.lease_id].state == "committed"

    async def test_reserve_denied_without_quota(self, fake_clock):
        storage = MemoryStorage()
        limiter = FixedWindowRateLimiter(storage)
        item = RateLimitItemPerMinute(10)

        assert await limiter.reserve(item, cost=11) is None
        lease = await limiter.reserve(item, cost=10)
        assert lease is not None
        assert await limiter.reserve(item, cost=1) is None

    async def test_commit_idempotent_and_conflicting_result(self, fake_clock):
        storage = MemoryStorage()
        limiter = FixedWindowRateLimiter(storage)
        item = RateLimitItemPerMinute(10)

        lease = await limiter.reserve(item, cost=6)
        await limiter.commit(lease, 4)
        await limiter.commit(lease, 4)
        assert (await limiter.get_window_stats(item)).remaining == 6

        with pytest.raises(LeaseConflictError):
            await limiter.commit(lease, 5)
        with pytest.raises(LeaseConflictError):
            await limiter.cancel(lease)
        assert (await limiter.get_window_stats(item)).remaining == 6

    async def test_commit_cost_exceeding_reservation(self, fake_clock):
        storage = MemoryStorage()
        limiter = FixedWindowRateLimiter(storage)
        item = RateLimitItemPerMinute(10)

        lease = await limiter.reserve(item, cost=3)
        with pytest.raises(ValueError):
            await limiter.commit(lease, 4)

    async def test_cancel_releases_reservation(self, fake_clock):
        storage = MemoryStorage()
        limiter = FixedWindowRateLimiter(storage)
        item = RateLimitItemPerMinute(10)

        lease = await limiter.reserve(item, cost=6)
        assert (await limiter.get_window_stats(item)).remaining == 4

        await limiter.cancel(lease)
        await limiter.cancel(lease)
        assert (await limiter.get_window_stats(item)).remaining == 10
        assert storage.leases[lease.lease_id].state == "cancelled"

        with pytest.raises(LeaseConflictError):
            await limiter.commit(lease, 1)

    async def test_lease_expiry_releases_reservation(self, fake_clock):
        storage = MemoryStorage()
        limiter = FixedWindowRateLimiter(storage)
        item = RateLimitItemPerMinute(10)

        lease = await limiter.reserve(item, cost=10, ttl=5)
        assert (await limiter.get_window_stats(item)).remaining == 0

        fake_clock.advance(6)
        assert await wait_until(
            lambda: storage.storage.get(item.key_for(), 0) == 0
        )
        assert await wait_until(lambda: lease.lease_id not in storage.leases)

        with pytest.raises(LeaseExpiredError):
            await limiter.commit(lease, 1)

    async def test_late_cancel_does_not_credit_new_window(self, fake_clock):
        storage = MemoryStorage()
        limiter = FixedWindowRateLimiter(storage)
        item = RateLimitItemPerSecond(10, 10)

        lease = await limiter.reserve(item, cost=5, ttl=30)
        assert (await limiter.get_window_stats(item)).remaining == 5

        fake_clock.advance(11)
        assert await limiter.hit(item)
        assert (await limiter.get_window_stats(item)).remaining == 9

        with pytest.raises(LeaseExpiredError):
            await limiter.commit(lease, 2)
        assert (await limiter.get_window_stats(item)).remaining == 9

        await limiter.cancel(lease)
        assert (await limiter.get_window_stats(item)).remaining == 9

    async def test_competing_committers_same_result(self, fake_clock):
        storage = MemoryStorage()
        limiter = FixedWindowRateLimiter(storage)
        item = RateLimitItemPerMinute(10)
        lease = await limiter.reserve(item, cost=10)

        start = asyncio.Event()

        async def committer():
            await start.wait()
            await limiter.commit(lease, 4)

        tasks = [asyncio.create_task(committer()) for _ in range(2)]
        start.set()
        await asyncio.gather(*tasks)

        assert (await limiter.get_window_stats(item)).remaining == 6
        assert storage.leases[lease.lease_id].state == "committed"

    async def test_competing_committers_different_result(self, fake_clock):
        storage = MemoryStorage()
        limiter = FixedWindowRateLimiter(storage)
        item = RateLimitItemPerMinute(10)
        lease = await limiter.reserve(item, cost=10)

        start = asyncio.Event()
        conflicts = []

        async def committer(cost):
            await start.wait()
            try:
                await limiter.commit(lease, cost)
            except LeaseConflictError:
                conflicts.append(cost)

        tasks = [
            asyncio.create_task(committer(4)),
            asyncio.create_task(committer(6)),
        ]
        start.set()
        await asyncio.gather(*tasks)

        assert len(conflicts) == 1
        committed = storage.leases[lease.lease_id].committed_cost
        assert committed in (4, 6)
        assert (await limiter.get_window_stats(item)).remaining == 10 - committed

    async def test_lease_records_are_reclaimed(self, fake_clock):
        storage = MemoryStorage()
        limiter = FixedWindowRateLimiter(storage)
        item = RateLimitItemPerMinute(100)

        leases = [await limiter.reserve(item, cost=1, ttl=5) for _ in range(10)]
        await limiter.commit(leases[0], 1)
        await limiter.cancel(leases[1])
        assert len(storage.leases) == 10

        fake_clock.advance(6)
        assert await wait_until(lambda: not storage.leases)
        assert (await limiter.get_window_stats(item)).remaining == 99

    async def test_reservations_not_supported_by_other_strategies(self, fake_clock):
        storage = MemoryStorage()
        item = RateLimitItemPerMinute(10)

        for strategy in (MovingWindowRateLimiter, SlidingWindowCounterRateLimiter):
            limiter = strategy(storage)
            with pytest.raises(NotImplementedError):
                await limiter.reserve(item)
            with pytest.raises(NotImplementedError):
                await limiter.commit(None, 1)
            with pytest.raises(NotImplementedError):
                await limiter.cancel(None)

    async def test_hit_and_test_unaffected_by_reservations(self, fake_clock):
        storage = MemoryStorage()
        limiter = FixedWindowRateLimiter(storage)
        item = RateLimitItemPerMinute(10)

        lease = await limiter.reserve(item, cost=5)
        assert all([await limiter.hit(item) for _ in range(5)])
        assert not await limiter.test(item)

        await limiter.cancel(lease)
        assert all([await limiter.hit(item) for _ in range(5)])
        assert not await limiter.hit(item)
