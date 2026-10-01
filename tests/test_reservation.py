from __future__ import annotations

import threading
import time

import pytest

import limits.storage.memory
from limits import errors
from limits.limits import RateLimitItemPerSecond
from limits.storage import storage_from_string
from limits.storage.base import Lease, Storage
from limits.strategies import (
    FixedWindowRateLimiter,
    MovingWindowRateLimiter,
    SlidingWindowCounterRateLimiter,
)


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
    monkeypatch.setattr(limits.storage.memory, "time", clock)
    return clock


@pytest.fixture
def storage(clock):
    return storage_from_string("memory://")


@pytest.fixture
def limiter(storage):
    return FixedWindowRateLimiter(storage)


def wait_for_reclaim(storage, lease_id, timeout: float = 2.0) -> None:
    deadline = time.time() + timeout
    while lease_id in storage.leases and time.time() < deadline:
        time.sleep(0.01)


class TestFixedWindowReservation:
    def test_reserve_commit_settles_actual_cost(self, storage, limiter):
        limit = RateLimitItemPerSecond(10, 60)
        lease = limiter.reserve(limit, cost=8)
        assert lease is not None
        assert lease.amount == 8
        assert lease.key == limit.key_for()
        assert limiter.get_window_stats(limit).remaining == 2

        limiter.commit(lease, 5)
        assert limiter.get_window_stats(limit).remaining == 5
        record = storage.leases[lease.lease_id]
        assert record.state == "committed"
        assert record.settled_cost == 5

    def test_reserve_denied_when_insufficient(self, limiter):
        limit = RateLimitItemPerSecond(10, 60)
        assert limiter.reserve(limit, cost=11) is None
        assert limiter.reserve(limit, cost=8) is not None
        assert limiter.reserve(limit, cost=3) is None
        assert limiter.reserve(limit, cost=2) is not None

    def test_commit_idempotent_and_conflicting_result(self, limiter):
        limit = RateLimitItemPerSecond(10, 60)
        lease = limiter.reserve(limit, cost=8)

        limiter.commit(lease, 5)
        limiter.commit(lease, 5)
        assert limiter.get_window_stats(limit).remaining == 5

        with pytest.raises(errors.LeaseConflictError):
            limiter.commit(lease, 6)
        with pytest.raises(errors.LeaseConflictError):
            limiter.cancel(lease)

    def test_cancel_refunds_and_is_idempotent(self, storage, limiter):
        limit = RateLimitItemPerSecond(10, 60)
        lease = limiter.reserve(limit, cost=8)
        assert limiter.get_window_stats(limit).remaining == 2

        limiter.cancel(lease)
        limiter.cancel(lease)
        assert limiter.get_window_stats(limit).remaining == 10
        assert storage.leases[lease.lease_id].state == "cancelled"

        with pytest.raises(errors.LeaseConflictError):
            limiter.commit(lease, 1)

    def test_commit_cost_must_not_exceed_reservation(self, limiter):
        limit = RateLimitItemPerSecond(10, 60)
        lease = limiter.reserve(limit, cost=8)
        with pytest.raises(ValueError):
            limiter.commit(lease, 9)
        with pytest.raises(ValueError):
            limiter.commit(lease, -1)

    def test_commit_unknown_lease(self, limiter):
        limit = RateLimitItemPerSecond(10, 60)
        lease = Lease(
            lease_id="unknown",
            key=limit.key_for(),
            generation=0,
            amount=1,
            expires_at=0,
        )
        with pytest.raises(errors.LeaseExpiredError):
            limiter.commit(lease, 1)
        with pytest.raises(errors.LeaseExpiredError):
            limiter.cancel(lease)

    def test_lease_expiry_releases_reservation(self, clock, storage, limiter):
        limit = RateLimitItemPerSecond(10, 60)
        lease = limiter.reserve(limit, cost=8, lease_expiry=5)
        assert limiter.get_window_stats(limit).remaining == 2

        clock.advance(6)
        # trigger the background expiry and wait for the lease to be reclaimed
        limiter.reserve(limit, cost=1, lease_expiry=60)
        wait_for_reclaim(storage, lease.lease_id)

        assert lease.lease_id not in storage.leases
        assert limiter.get_window_stats(limit).remaining == 9

    def test_expired_lease_cannot_be_settled(self, clock, storage, limiter):
        limit = RateLimitItemPerSecond(10, 60)
        lease = limiter.reserve(limit, cost=8, lease_expiry=5)

        clock.advance(6)
        with pytest.raises(errors.LeaseExpiredError):
            limiter.commit(lease, 5)
        # the expired reservation was released back to the window
        assert limiter.get_window_stats(limit).remaining == 10
        with pytest.raises(errors.LeaseExpiredError):
            limiter.cancel(lease)
        assert lease.lease_id not in storage.leases

    def test_late_cancel_after_window_rotation_does_not_recharge(
        self, clock, storage, limiter
    ):
        limit = RateLimitItemPerSecond(10, 10)
        lease = limiter.reserve(limit, cost=8, lease_expiry=100)
        assert limiter.get_window_stats(limit).remaining == 2

        # cross the window boundary; the lease itself is still valid
        clock.advance(11)
        limiter.cancel(lease)
        assert storage.leases[lease.lease_id].state == "cancelled"

        # the new window must have the full quota: the late cancel
        # did not recharge it
        assert all(limiter.hit(limit) for _ in range(10))
        assert not limiter.hit(limit)

    def test_lease_expiry_after_window_rotation_does_not_recharge(
        self, clock, storage, limiter
    ):
        limit = RateLimitItemPerSecond(10, 10)
        lease = limiter.reserve(limit, cost=8, lease_expiry=15)

        # rotate the window and consume 1 from the new window
        clock.advance(11)
        assert limiter.hit(limit)

        # now let the lease expire; its release must not touch the new window
        clock.advance(5)
        limiter.reserve(limit, cost=1, lease_expiry=60)
        wait_for_reclaim(storage, lease.lease_id)

        assert lease.lease_id not in storage.leases
        assert limiter.get_window_stats(limit).remaining == 8

    def test_competing_committers(self, clock, storage, limiter):
        limit = RateLimitItemPerSecond(10, 60)
        lease = limiter.reserve(limit, cost=8)
        barrier = threading.Barrier(2)
        results = []

        def settle(cost):
            barrier.wait()
            try:
                limiter.commit(lease, cost)
                results.append(("ok", cost))
            except errors.LeaseConflictError:
                results.append(("conflict", cost))

        threads = [threading.Thread(target=settle, args=(cost,)) for cost in (5, 6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert sorted(outcome for outcome, _ in results) == ["conflict", "ok"]
        winner = [cost for outcome, cost in results if outcome == "ok"][0]
        assert limiter.get_window_stats(limit).remaining == 10 - winner
        record = storage.leases[lease.lease_id]
        assert record.state == "committed"
        assert record.settled_cost == winner

    def test_clear_drops_leases(self, storage, limiter):
        limit = RateLimitItemPerSecond(10, 60)
        lease = limiter.reserve(limit, cost=8)
        limiter.clear(limit)
        assert lease.lease_id not in storage.leases
        with pytest.raises(errors.LeaseExpiredError):
            limiter.cancel(lease)

    def test_hit_and_test_unchanged(self, clock, limiter):
        limit = RateLimitItemPerSecond(2, 10)
        assert limiter.hit(limit)
        assert limiter.test(limit)
        assert limiter.hit(limit)
        assert not limiter.test(limit)
        assert not limiter.hit(limit)
        clock.advance(11)
        assert limiter.test(limit)
        assert limiter.hit(limit)

    def test_reservations_share_quota_with_hits(self, limiter):
        limit = RateLimitItemPerSecond(10, 60)
        assert limiter.hit(limit, cost=4)
        lease = limiter.reserve(limit, cost=4)
        assert limiter.get_window_stats(limit).remaining == 2
        limiter.commit(lease, 1)
        assert limiter.get_window_stats(limit).remaining == 5


class BareStorage(Storage):
    STORAGE_SCHEME = None

    @property
    def base_exceptions(self):
        return ValueError

    def incr(self, key, expiry, amount=1):
        return amount

    def get(self, key):
        return 0

    def get_expiry(self, key):
        return 0.0

    def check(self):
        return True

    def reset(self):
        return None

    def clear(self, key):
        pass


class TestReservationSupport:
    def test_fixed_window_without_reservation_support(self):
        limiter = FixedWindowRateLimiter(BareStorage())
        limit = RateLimitItemPerSecond(10, 60)
        lease = Lease("x", "k", 0, 1, 0)
        with pytest.raises(NotImplementedError):
            limiter.reserve(limit)
        with pytest.raises(NotImplementedError):
            limiter.commit(lease, 1)
        with pytest.raises(NotImplementedError):
            limiter.cancel(lease)

    def test_other_strategies_not_supported(self, storage):
        limit = RateLimitItemPerSecond(10, 60)
        for strategy in (MovingWindowRateLimiter, SlidingWindowCounterRateLimiter):
            limiter = strategy(storage)
            lease = Lease("x", "k", 0, 1, 0)
            with pytest.raises(NotImplementedError):
                limiter.reserve(limit)
            with pytest.raises(NotImplementedError):
                limiter.commit(lease, 1)
            with pytest.raises(NotImplementedError):
                limiter.cancel(lease)
