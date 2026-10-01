from __future__ import annotations

import threading
import time

import pytest

from limits.errors import LeaseConflictError, LeaseExpiredError
from limits.limits import RateLimitItemPerMinute, RateLimitItemPerSecond
from limits.storage import MemoryStorage
from limits.strategies import (
    FixedWindowRateLimiter,
    MovingWindowRateLimiter,
    SlidingWindowCounterRateLimiter,
)


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


def wait_until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


@pytest.mark.memory
class TestFixedWindowReservation:
    def test_reserve_commit_settles_actual_cost(self, fake_clock):
        storage = MemoryStorage()
        limiter = FixedWindowRateLimiter(storage)
        item = RateLimitItemPerMinute(10)

        lease = limiter.reserve(item, cost=8)
        assert lease is not None
        assert limiter.get_window_stats(item).remaining == 2

        limiter.commit(lease, 3)
        assert limiter.get_window_stats(item).remaining == 7
        assert storage.leases[lease.lease_id].state == "committed"

    def test_reserve_denied_without_quota(self, fake_clock):
        storage = MemoryStorage()
        limiter = FixedWindowRateLimiter(storage)
        item = RateLimitItemPerMinute(10)

        assert limiter.reserve(item, cost=11) is None
        lease = limiter.reserve(item, cost=10)
        assert lease is not None
        assert limiter.reserve(item, cost=1) is None

    def test_commit_idempotent_and_conflicting_result(self, fake_clock):
        storage = MemoryStorage()
        limiter = FixedWindowRateLimiter(storage)
        item = RateLimitItemPerMinute(10)

        lease = limiter.reserve(item, cost=6)
        limiter.commit(lease, 4)
        limiter.commit(lease, 4)
        assert limiter.get_window_stats(item).remaining == 6

        with pytest.raises(LeaseConflictError):
            limiter.commit(lease, 5)
        with pytest.raises(LeaseConflictError):
            limiter.cancel(lease)
        assert limiter.get_window_stats(item).remaining == 6

    def test_commit_cost_exceeding_reservation(self, fake_clock):
        storage = MemoryStorage()
        limiter = FixedWindowRateLimiter(storage)
        item = RateLimitItemPerMinute(10)

        lease = limiter.reserve(item, cost=3)
        with pytest.raises(ValueError):
            limiter.commit(lease, 4)

    def test_cancel_releases_reservation(self, fake_clock):
        storage = MemoryStorage()
        limiter = FixedWindowRateLimiter(storage)
        item = RateLimitItemPerMinute(10)

        lease = limiter.reserve(item, cost=6)
        assert limiter.get_window_stats(item).remaining == 4

        limiter.cancel(lease)
        limiter.cancel(lease)
        assert limiter.get_window_stats(item).remaining == 10
        assert storage.leases[lease.lease_id].state == "cancelled"

        with pytest.raises(LeaseConflictError):
            limiter.commit(lease, 1)

    def test_lease_expiry_releases_reservation(self, fake_clock):
        storage = MemoryStorage()
        limiter = FixedWindowRateLimiter(storage)
        item = RateLimitItemPerMinute(10)

        lease = limiter.reserve(item, cost=10, ttl=5)
        assert limiter.get_window_stats(item).remaining == 0

        fake_clock.advance(6)
        assert wait_until(lambda: limiter.get_window_stats(item).remaining == 10)
        assert wait_until(lambda: lease.lease_id not in storage.leases)

        with pytest.raises(LeaseExpiredError):
            limiter.commit(lease, 1)

    def test_late_cancel_does_not_credit_new_window(self, fake_clock):
        storage = MemoryStorage()
        limiter = FixedWindowRateLimiter(storage)
        item = RateLimitItemPerSecond(10, 10)

        lease = limiter.reserve(item, cost=5, ttl=30)
        assert limiter.get_window_stats(item).remaining == 5

        fake_clock.advance(11)
        assert limiter.hit(item)
        assert limiter.get_window_stats(item).remaining == 9

        with pytest.raises(LeaseExpiredError):
            limiter.commit(lease, 2)
        assert limiter.get_window_stats(item).remaining == 9

        limiter.cancel(lease)
        assert limiter.get_window_stats(item).remaining == 9

    def test_competing_committers_same_result(self, fake_clock):
        storage = MemoryStorage()
        limiter = FixedWindowRateLimiter(storage)
        item = RateLimitItemPerMinute(10)
        lease = limiter.reserve(item, cost=10)

        barrier = threading.Barrier(2)
        errors = []

        def committer():
            try:
                barrier.wait(timeout=5)
                limiter.commit(lease, 4)
            except Exception as exc:  # noqa
                errors.append(exc)

        threads = [threading.Thread(target=committer) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert errors == []
        assert limiter.get_window_stats(item).remaining == 6
        assert storage.leases[lease.lease_id].state == "committed"

    def test_competing_committers_different_result(self, fake_clock):
        storage = MemoryStorage()
        limiter = FixedWindowRateLimiter(storage)
        item = RateLimitItemPerMinute(10)
        lease = limiter.reserve(item, cost=10)

        barrier = threading.Barrier(2)
        conflicts = []

        def committer(cost):
            try:
                barrier.wait(timeout=5)
                limiter.commit(lease, cost)
            except LeaseConflictError:
                conflicts.append(cost)

        threads = [
            threading.Thread(target=committer, args=(4,)),
            threading.Thread(target=committer, args=(6,)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(conflicts) == 1
        committed = storage.leases[lease.lease_id].committed_cost
        assert committed in (4, 6)
        assert limiter.get_window_stats(item).remaining == 10 - committed

    def test_lease_records_are_reclaimed(self, fake_clock):
        storage = MemoryStorage()
        limiter = FixedWindowRateLimiter(storage)
        item = RateLimitItemPerMinute(100)

        leases = [limiter.reserve(item, cost=1, ttl=5) for _ in range(10)]
        limiter.commit(leases[0], 1)
        limiter.cancel(leases[1])
        assert len(storage.leases) == 10

        fake_clock.advance(6)
        assert wait_until(lambda: not storage.leases)
        assert limiter.get_window_stats(item).remaining == 99

    def test_reservations_not_supported_by_other_strategies(self, fake_clock):
        storage = MemoryStorage()
        item = RateLimitItemPerMinute(10)

        for strategy in (MovingWindowRateLimiter, SlidingWindowCounterRateLimiter):
            limiter = strategy(storage)
            with pytest.raises(NotImplementedError):
                limiter.reserve(item)
            with pytest.raises(NotImplementedError):
                limiter.commit(None, 1)
            with pytest.raises(NotImplementedError):
                limiter.cancel(None)

    def test_hit_and_test_unaffected_by_reservations(self, fake_clock):
        storage = MemoryStorage()
        limiter = FixedWindowRateLimiter(storage)
        item = RateLimitItemPerMinute(10)

        lease = limiter.reserve(item, cost=5)
        assert all(limiter.hit(item) for _ in range(5))
        assert not limiter.test(item)

        limiter.cancel(lease)
        assert all(limiter.hit(item) for _ in range(5))
        assert not limiter.hit(item)
