from __future__ import annotations

import asyncio
import time

import pytest

import limits.aio.storage.memory
from limits import errors
from limits.aio.strategies import (
    FixedWindowRateLimiter,
    MovingWindowRateLimiter,
    SlidingWindowCounterRateLimiter,
)
from limits.limits import RateLimitItemPerSecond
from limits.storage import storage_from_string
from limits.storage.base import Lease


class FakeClock:
    def __init__(self, start: float = 1_000_000.0) -> None:
        self.now = start

    def time(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(limits.aio.storage.memory, "time", clock)
    return clock


@pytest.fixture
def storage(clock):
    return storage_from_string("async+memory://")


@pytest.fixture
def limiter(storage):
    return FixedWindowRateLimiter(storage)


async def wait_for_reclaim(storage, lease_id, timeout: float = 2.0) -> None:
    deadline = time.time() + timeout
    while lease_id in storage.leases and time.time() < deadline:
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
class TestAsyncFixedWindowReservation:
    async def test_reserve_commit_settles_actual_cost(self, storage, limiter):
        limit = RateLimitItemPerSecond(10, 60)
        lease = await limiter.reserve(limit, cost=8)
        assert lease is not None
        assert lease.amount == 8
        assert (await limiter.get_window_stats(limit)).remaining == 2

        await limiter.commit(lease, 5)
        assert (await limiter.get_window_stats(limit)).remaining == 5
        record = storage.leases[lease.lease_id]
        assert record.state == "committed"
        assert record.settled_cost == 5

    async def test_reserve_denied_when_insufficient(self, limiter):
        limit = RateLimitItemPerSecond(10, 60)
        assert await limiter.reserve(limit, cost=11) is None
        assert await limiter.reserve(limit, cost=8) is not None
        assert await limiter.reserve(limit, cost=3) is None

    async def test_commit_idempotent_and_conflicting_result(self, limiter):
        limit = RateLimitItemPerSecond(10, 60)
        lease = await limiter.reserve(limit, cost=8)

        await limiter.commit(lease, 5)
        await limiter.commit(lease, 5)
        assert (await limiter.get_window_stats(limit)).remaining == 5

        with pytest.raises(errors.LeaseConflictError):
            await limiter.commit(lease, 6)
        with pytest.raises(errors.LeaseConflictError):
            await limiter.cancel(lease)

    async def test_cancel_refunds_and_is_idempotent(self, storage, limiter):
        limit = RateLimitItemPerSecond(10, 60)
        lease = await limiter.reserve(limit, cost=8)

        await limiter.cancel(lease)
        await limiter.cancel(lease)
        assert (await limiter.get_window_stats(limit)).remaining == 10
        assert storage.leases[lease.lease_id].state == "cancelled"

        with pytest.raises(errors.LeaseConflictError):
            await limiter.commit(lease, 1)

    async def test_commit_cost_must_not_exceed_reservation(self, limiter):
        limit = RateLimitItemPerSecond(10, 60)
        lease = await limiter.reserve(limit, cost=8)
        with pytest.raises(ValueError):
            await limiter.commit(lease, 9)

    async def test_commit_unknown_lease(self, limiter):
        limit = RateLimitItemPerSecond(10, 60)
        lease = Lease(
            lease_id="unknown",
            key=limit.key_for(),
            generation=0,
            amount=1,
            expires_at=0,
        )
        with pytest.raises(errors.LeaseExpiredError):
            await limiter.commit(lease, 1)
        with pytest.raises(errors.LeaseExpiredError):
            await limiter.cancel(lease)

    async def test_lease_expiry_releases_reservation(self, clock, storage, limiter):
        limit = RateLimitItemPerSecond(10, 60)
        lease = await limiter.reserve(limit, cost=8, lease_expiry=5)
        assert (await limiter.get_window_stats(limit)).remaining == 2

        clock.advance(6)
        # trigger the background expiry and wait for the lease to be reclaimed
        await limiter.reserve(limit, cost=1, lease_expiry=60)
        await wait_for_reclaim(storage, lease.lease_id)

        assert lease.lease_id not in storage.leases
        assert (await limiter.get_window_stats(limit)).remaining == 9

    async def test_expired_lease_cannot_be_settled(self, clock, storage, limiter):
        limit = RateLimitItemPerSecond(10, 60)
        lease = await limiter.reserve(limit, cost=8, lease_expiry=5)

        clock.advance(6)
        with pytest.raises(errors.LeaseExpiredError):
            await limiter.commit(lease, 5)
        assert (await limiter.get_window_stats(limit)).remaining == 10
        with pytest.raises(errors.LeaseExpiredError):
            await limiter.cancel(lease)
        assert lease.lease_id not in storage.leases

    async def test_late_cancel_after_window_rotation_does_not_recharge(
        self, clock, storage, limiter
    ):
        limit = RateLimitItemPerSecond(10, 10)
        lease = await limiter.reserve(limit, cost=8, lease_expiry=100)

        clock.advance(11)
        await limiter.cancel(lease)
        assert storage.leases[lease.lease_id].state == "cancelled"

        assert all([await limiter.hit(limit) for _ in range(10)])
        assert not await limiter.hit(limit)

    async def test_lease_expiry_after_window_rotation_does_not_recharge(
        self, clock, storage, limiter
    ):
        limit = RateLimitItemPerSecond(10, 10)
        lease = await limiter.reserve(limit, cost=8, lease_expiry=15)

        clock.advance(11)
        assert await limiter.hit(limit)

        clock.advance(5)
        await limiter.reserve(limit, cost=1, lease_expiry=60)
        await wait_for_reclaim(storage, lease.lease_id)

        assert lease.lease_id not in storage.leases
        assert (await limiter.get_window_stats(limit)).remaining == 8

    async def test_competing_committers(self, clock, storage, limiter):
        limit = RateLimitItemPerSecond(10, 60)
        lease = await limiter.reserve(limit, cost=8)
        results = []

        async def settle(cost):
            try:
                await limiter.commit(lease, cost)
                results.append(("ok", cost))
            except errors.LeaseConflictError:
                results.append(("conflict", cost))

        await asyncio.gather(settle(5), settle(6))

        assert sorted(outcome for outcome, _ in results) == ["conflict", "ok"]
        winner = [cost for outcome, cost in results if outcome == "ok"][0]
        assert (await limiter.get_window_stats(limit)).remaining == 10 - winner
        record = storage.leases[lease.lease_id]
        assert record.state == "committed"
        assert record.settled_cost == winner

    async def test_hit_and_test_unchanged(self, clock, limiter):
        limit = RateLimitItemPerSecond(2, 10)
        assert await limiter.hit(limit)
        assert await limiter.test(limit)
        assert await limiter.hit(limit)
        assert not await limiter.test(limit)
        assert not await limiter.hit(limit)
        clock.advance(11)
        assert await limiter.test(limit)
        assert await limiter.hit(limit)

    async def test_other_strategies_not_supported(self, storage):
        limit = RateLimitItemPerSecond(10, 60)
        for strategy in (MovingWindowRateLimiter, SlidingWindowCounterRateLimiter):
            limiter = strategy(storage)
            lease = Lease("x", "k", 0, 1, 0)
            with pytest.raises(NotImplementedError):
                await limiter.reserve(limit)
            with pytest.raises(NotImplementedError):
                await limiter.commit(lease, 1)
            with pytest.raises(NotImplementedError):
                await limiter.cancel(lease)
