from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from limits.limits import RateLimitItemPerSecond
from limits.storage import MemoryStorage
from limits.strategies import GCRARateLimiter
from tests.utils import ManualClock


@pytest.fixture
def clock():
    return ManualClock()


@pytest.fixture
def limiter(clock):
    return GCRARateLimiter(MemoryStorage(clock=clock))


@pytest.fixture
def limit():
    # 5 requests per 10 seconds: emission interval 2s, burst capacity 5
    return RateLimitItemPerSecond(5, 10)


class TestGCRAParameters:
    def test_rate_and_burst_definitions(self, limiter, limit):
        assert limiter.get_emission_interval(limit) == 2.0
        assert limiter.get_burst_capacity(limit) == 5
        assert limiter.get_burst_tolerance(limit) == 8.0

    def test_unsupported_storage(self):
        class Dummy:
            pass

        with pytest.raises(NotImplementedError):
            GCRARateLimiter(Dummy())


class TestGCRA:
    def test_burst_then_reject(self, limiter, limit):
        assert all(limiter.hit(limit) for _ in range(5))
        assert not limiter.hit(limit)

    def test_burst_capacity_recovers_at_emission_interval(self, clock, limiter, limit):
        assert all(limiter.hit(limit) for _ in range(5))
        clock.advance(2)
        assert limiter.hit(limit)
        assert not limiter.hit(limit)
        clock.advance(8)
        assert all(limiter.hit(limit) for _ in range(4))
        assert not limiter.hit(limit)

    def test_steady_rate_always_allowed(self, clock, limiter, limit):
        for _ in range(20):
            clock.advance(2)
            assert limiter.hit(limit)

    def test_slower_than_steady_rate_allowed(self, clock, limiter, limit):
        for _ in range(20):
            clock.advance(3)
            assert limiter.hit(limit)

    def test_cost(self, clock, limiter, limit):
        assert limiter.hit(limit, cost=3)
        assert not limiter.hit(limit, cost=3)
        assert limiter.hit(limit, cost=2)
        assert not limiter.hit(limit)

    def test_cost_exceeding_burst_capacity_rejected_immediately(
        self, clock, limiter, limit
    ):
        assert not limiter.hit(limit, cost=6)
        # the rejected hit did not consume anything
        assert limiter.hit(limit, cost=5)
        assert not limiter.hit(limit)

    def test_cost_equal_to_burst_capacity_allowed_on_empty_state(
        self, clock, limiter, limit
    ):
        assert limiter.hit(limit, cost=5)
        assert not limiter.hit(limit)
        clock.advance(10)
        assert limiter.hit(limit, cost=5)

    def test_rejected_hit_does_not_advance_theoretical_arrival_time(
        self, clock, limiter, limit
    ):
        assert all(limiter.hit(limit) for _ in range(5))
        stats_before = limiter.get_window_stats(limit)
        assert not limiter.hit(limit)
        assert not limiter.hit(limit, cost=2)
        stats_after = limiter.get_window_stats(limit)
        assert stats_after == stats_before
        clock.advance(2)
        assert limiter.hit(limit)

    def test_test_does_not_consume(self, clock, limiter, limit):
        assert limiter.test(limit)
        assert limiter.test(limit, cost=5)
        assert not limiter.test(limit, cost=6)
        # observations did not reserve any capacity
        assert all(limiter.hit(limit) for _ in range(5))
        assert not limiter.test(limit)
        assert not limiter.hit(limit)

    def test_exact_boundary(self, clock, limiter, limit):
        assert all(limiter.hit(limit) for _ in range(5))
        stats = limiter.get_window_stats(limit)
        assert stats.reset_time == 1002.0
        # before the earliest retry: rejected
        clock.advance(1.5)
        assert not limiter.hit(limit)
        # exactly on the boundary: allowed
        clock.advance(0.5)
        assert limiter.hit(limit)

    def test_window_stats_match_decisions(self, clock, limiter, limit):
        stats = limiter.get_window_stats(limit)
        assert stats.remaining == 5
        assert stats.reset_time == clock.now

        assert limiter.hit(limit, cost=2)
        stats = limiter.get_window_stats(limit)
        assert stats.remaining == 3
        # capacity is still available: a unit cost hit is allowed right now
        assert stats.reset_time == clock.now

        # the reported remaining matches the number of allowed hits
        assert all(limiter.hit(limit) for _ in range(stats.remaining))
        assert not limiter.hit(limit)
        stats = limiter.get_window_stats(limit)
        assert stats.remaining == 0
        # earliest retry is when one more unit fits, not a fixed window reset
        assert stats.reset_time == clock.now + 2

    def test_window_stats_reset_time_is_earliest_retry(self, clock, limiter, limit):
        assert all(limiter.hit(limit) for _ in range(5))
        stats = limiter.get_window_stats(limit)
        # theoretical arrival time is now + 10, burst tolerance 8
        assert stats.reset_time == clock.now + 2
        clock.advance(2)
        assert limiter.get_window_stats(limit).reset_time == clock.now

    def test_idle_state_is_released(self, clock, limiter, limit):
        storage = limiter.storage
        key = limit.key_for()
        assert limiter.hit(limit)
        assert key in storage.gcra
        # let the theoretical arrival time fully drain
        clock.advance(11)
        assert limiter.get_window_stats(limit).remaining == 5
        assert key not in storage.gcra
        # the full burst capacity is available again
        assert all(limiter.hit(limit) for _ in range(5))

    def test_clear(self, limiter, limit):
        assert all(limiter.hit(limit) for _ in range(5))
        limiter.clear(limit)
        assert limiter.get_window_stats(limit).remaining == 5
        assert all(limiter.hit(limit) for _ in range(5))

    def test_concurrent_hits_never_exceed_burst_capacity(
        self, clock, limiter, limit
    ):
        barrier = threading.Barrier(20)

        def hit():
            barrier.wait()
            return limiter.hit(limit)

        with ThreadPoolExecutor(max_workers=20) as executor:
            results = list(executor.map(lambda _: hit(), range(20)))

        assert sum(results) == 5
        assert limiter.get_window_stats(limit).remaining == 0

    def test_concurrent_hits_at_steady_rate(self, clock, limiter, limit):
        # exhaust the burst capacity first
        assert all(limiter.hit(limit) for _ in range(5))
        for _ in range(5):
            clock.advance(2)
            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(lambda _: limiter.hit(limit), range(4)))
            assert sum(results) == 1
