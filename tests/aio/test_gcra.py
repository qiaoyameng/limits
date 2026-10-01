from __future__ import annotations

import asyncio

import pytest

from limits.aio.storage import MemoryStorage as AsyncMemoryStorage
from limits.aio.strategies import GCRARateLimiter as AsyncGCRARateLimiter
from limits.limits import RateLimitItemPerSecond
from limits.storage import MemoryStorage
from limits.strategies import GCRARateLimiter
from tests.utils import ManualClock


@pytest.fixture
def clock():
    return ManualClock()


@pytest.fixture
def limiter(clock):
    return AsyncGCRARateLimiter(AsyncMemoryStorage(clock=clock))


@pytest.fixture
def limit():
    # 5 requests per 10 seconds: emission interval 2s, burst capacity 5
    return RateLimitItemPerSecond(5, 10)


class TestGCRAParameters:
    async def test_rate_and_burst_definitions(self, limiter, limit):
        assert limiter.get_emission_interval(limit) == 2.0
        assert limiter.get_burst_capacity(limit) == 5
        assert limiter.get_burst_tolerance(limit) == 8.0

    async def test_unsupported_storage(self):
        class Dummy:
            pass

        with pytest.raises(NotImplementedError):
            AsyncGCRARateLimiter(Dummy())


class TestGCRA:
    async def test_burst_then_reject(self, limiter, limit):
        assert all([await limiter.hit(limit) for _ in range(5)])
        assert not await limiter.hit(limit)

    async def test_burst_capacity_recovers_at_emission_interval(
        self, clock, limiter, limit
    ):
        assert all([await limiter.hit(limit) for _ in range(5)])
        clock.advance(2)
        assert await limiter.hit(limit)
        assert not await limiter.hit(limit)
        clock.advance(8)
        assert all([await limiter.hit(limit) for _ in range(4)])
        assert not await limiter.hit(limit)

    async def test_steady_rate_always_allowed(self, clock, limiter, limit):
        for _ in range(20):
            clock.advance(2)
            assert await limiter.hit(limit)

    async def test_cost(self, clock, limiter, limit):
        assert await limiter.hit(limit, cost=3)
        assert not await limiter.hit(limit, cost=3)
        assert await limiter.hit(limit, cost=2)
        assert not await limiter.hit(limit)

    async def test_cost_exceeding_burst_capacity_rejected_immediately(
        self, clock, limiter, limit
    ):
        assert not await limiter.hit(limit, cost=6)
        # the rejected hit did not consume anything
        assert await limiter.hit(limit, cost=5)
        assert not await limiter.hit(limit)

    async def test_rejected_hit_does_not_advance_theoretical_arrival_time(
        self, clock, limiter, limit
    ):
        assert all([await limiter.hit(limit) for _ in range(5)])
        stats_before = await limiter.get_window_stats(limit)
        assert not await limiter.hit(limit)
        assert not await limiter.hit(limit, cost=2)
        stats_after = await limiter.get_window_stats(limit)
        assert stats_after == stats_before
        clock.advance(2)
        assert await limiter.hit(limit)

    async def test_test_does_not_consume(self, clock, limiter, limit):
        assert await limiter.test(limit)
        assert await limiter.test(limit, cost=5)
        assert not await limiter.test(limit, cost=6)
        # observations did not reserve any capacity
        assert all([await limiter.hit(limit) for _ in range(5)])
        assert not await limiter.test(limit)
        assert not await limiter.hit(limit)

    async def test_exact_boundary(self, clock, limiter, limit):
        assert all([await limiter.hit(limit) for _ in range(5)])
        stats = await limiter.get_window_stats(limit)
        assert stats.reset_time == 1002.0
        # before the earliest retry: rejected
        clock.advance(1.5)
        assert not await limiter.hit(limit)
        # exactly on the boundary: allowed
        clock.advance(0.5)
        assert await limiter.hit(limit)

    async def test_window_stats_match_decisions(self, clock, limiter, limit):
        stats = await limiter.get_window_stats(limit)
        assert stats.remaining == 5
        assert stats.reset_time == clock.now

        assert await limiter.hit(limit, cost=2)
        stats = await limiter.get_window_stats(limit)
        assert stats.remaining == 3
        assert stats.reset_time == clock.now

        # the reported remaining matches the number of allowed hits
        assert all([await limiter.hit(limit) for _ in range(stats.remaining)])
        assert not await limiter.hit(limit)
        stats = await limiter.get_window_stats(limit)
        assert stats.remaining == 0
        # earliest retry is when one more unit fits, not a fixed window reset
        assert stats.reset_time == clock.now + 2

    async def test_idle_state_is_released(self, clock, limiter, limit):
        storage = limiter.storage
        key = limit.key_for()
        assert await limiter.hit(limit)
        assert key in storage.gcra
        # let the theoretical arrival time fully drain
        clock.advance(11)
        assert (await limiter.get_window_stats(limit)).remaining == 5
        assert key not in storage.gcra
        # the full burst capacity is available again
        assert all([await limiter.hit(limit) for _ in range(5)])

    async def test_clear(self, limiter, limit):
        assert all([await limiter.hit(limit) for _ in range(5)])
        await limiter.clear(limit)
        assert (await limiter.get_window_stats(limit)).remaining == 5
        assert all([await limiter.hit(limit) for _ in range(5)])

    async def test_concurrent_hits_never_exceed_burst_capacity(
        self, clock, limiter, limit
    ):
        results = await asyncio.gather(*[limiter.hit(limit) for _ in range(20)])
        assert sum(results) == 5
        assert (await limiter.get_window_stats(limit)).remaining == 0

    async def test_concurrent_hits_at_steady_rate(self, clock, limiter, limit):
        # exhaust the burst capacity first
        assert all([await limiter.hit(limit) for _ in range(5)])
        for _ in range(5):
            clock.advance(2)
            results = await asyncio.gather(*[limiter.hit(limit) for _ in range(4)])
            assert sum(results) == 1


class TestSyncAsyncEquivalence:
    """
    The same input sequence must produce the same decisions
    with the synchronous and the asynchronous APIs.
    """

    SCRIPT = [
        ("hit", 1),
        ("hit", 1),
        ("advance", 1),
        ("hit", 2),
        ("test", 1),
        ("hit", 1),
        ("hit", 3),
        ("advance", 4),
        ("hit", 1),
        ("test", 2),
        ("hit", 2),
        ("advance", 0.5),
        ("hit", 1),
        ("advance", 10),
        ("hit", 5),
        ("hit", 6),
        ("test", 5),
        ("stats", None),
        ("advance", 3),
        ("hit", 1),
        ("stats", None),
    ]

    def run_sync(self, clock, limit):
        limiter = GCRARateLimiter(MemoryStorage(clock=clock))
        results = []
        for op, value in self.SCRIPT:
            if op == "advance":
                clock.advance(value)
            elif op == "hit":
                results.append(limiter.hit(limit, cost=value))
            elif op == "test":
                results.append(limiter.test(limit, cost=value))
            else:
                results.append(limiter.get_window_stats(limit))
        return results

    async def run_async(self, clock, limit):
        limiter = AsyncGCRARateLimiter(AsyncMemoryStorage(clock=clock))
        results = []
        for op, value in self.SCRIPT:
            if op == "advance":
                clock.advance(value)
            elif op == "hit":
                results.append(await limiter.hit(limit, cost=value))
            elif op == "test":
                results.append(await limiter.test(limit, cost=value))
            else:
                results.append(await limiter.get_window_stats(limit))
        return results

    async def test_same_sequence_same_results(self):
        limit = RateLimitItemPerSecond(5, 10)
        sync_results = self.run_sync(ManualClock(), limit)
        async_results = await self.run_async(ManualClock(), limit)
        assert sync_results == async_results
