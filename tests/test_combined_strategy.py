from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from limits.combined import CombinedRateLimitRequest as Req
from limits.errors import ConfigurationError
from limits.limits import RateLimitItemPerMinute, RateLimitItemPerSecond
from limits.storage import MemoryStorage
from limits.storage.base import Storage
from limits.strategies import CombinedRateLimiter


class FixedClock:
    def __init__(self, value: float = 1000.0):
        self.value = value

    def __call__(self) -> float:
        return self.value


class OneShotClock(FixedClock):
    def __init__(self, value: float = 1000.0):
        super().__init__(value)
        self.calls = 0

    def __call__(self) -> float:
        self.calls += 1
        if self.calls > 1:
            raise AssertionError("group must be evaluated at one timestamp")
        return self.value


class SteppedClock:
    def __init__(self, first: float, later: float):
        self.first = first
        self.later = later
        self.calls = 0

    def __call__(self) -> float:
        self.calls += 1
        return self.first if self.calls == 1 else self.later

    @property
    def value(self):
        return self.first


def fixed(item, *identifiers, cost=1):
    return Req.fixed_window(item, *identifiers, cost=cost)


def sliding(item, *identifiers, cost=1):
    return Req.sliding_window_counter(item, *identifiers, cost=cost)


def fill(storage, key, expiry, amount, now):
    storage.storage[key] = amount
    storage.expirations[key] = now + expiry


def test_group_success_consumes_all_limits():
    clock = FixedClock()
    storage = MemoryStorage(clock=clock)
    limiter = CombinedRateLimiter(storage)
    user = RateLimitItemPerMinute(2)
    tenant = RateLimitItemPerMinute(1)

    result = limiter.hit_many(
        [fixed(user, "user"), fixed(tenant, "tenant")]
    )

    assert result.allowed
    assert result.rejected == ()
    assert result.retry_at == clock.value
    assert storage.get(user.key_for("user")) == 1
    assert storage.get(tenant.key_for("tenant")) == 1


def test_group_failure_consumes_nothing():
    clock = FixedClock()
    storage = MemoryStorage(clock=clock)
    limiter = CombinedRateLimiter(storage)
    user = RateLimitItemPerMinute(2)
    tenant = RateLimitItemPerMinute(1)
    user_request = fixed(user, "user")
    tenant_request = fixed(tenant, "tenant")
    fill(storage, user.key_for("user"), 60, 1, clock.value)
    fill(storage, tenant.key_for("tenant"), 60, 1, clock.value)

    result = limiter.hit_many([user_request, tenant_request])

    assert not result
    assert result.rejected == (tenant_request,)
    assert result.retry_at == clock.value + 60
    assert storage.get(user.key_for("user")) == 1
    assert storage.get(tenant.key_for("tenant")) == 1


def test_earliest_retry_is_reported():
    clock = FixedClock()
    storage = MemoryStorage(clock=clock)
    limiter = CombinedRateLimiter(storage)
    slow = RateLimitItemPerMinute(1)
    fast = RateLimitItemPerSecond(1)
    fill(storage, slow.key_for("slow"), 60, 1, clock.value)
    fill(storage, fast.key_for("fast"), 1, 1, clock.value)

    result = limiter.hit_many([fixed(slow, "slow"), fixed(fast, "fast")])

    assert result.rejected == (fixed(slow, "slow"), fixed(fast, "fast"))
    assert result.retry_at == pytest.approx(clock.value + 1)


def test_duplicate_storage_keys_merge_cost():
    clock = FixedClock()
    storage = MemoryStorage(clock=clock)
    limiter = CombinedRateLimiter(storage)
    limit = RateLimitItemPerMinute(2)
    requests = [fixed(limit, "same"), fixed(limit, "same")]

    assert limiter.hit_many(requests).allowed
    assert storage.storage[limit.key_for("same")] == 2
    assert not limiter.hit_many(requests)
    assert storage.storage[limit.key_for("same")] == 2


def test_duplicate_storage_keys_with_different_limits_are_rejected():
    storage = MemoryStorage(clock=FixedClock())
    limiter = CombinedRateLimiter(storage)

    with pytest.raises(ConfigurationError):
        limiter.hit_many(
            [
                Req.fixed_window(RateLimitItemPerMinute(2), key="same"),
                Req.fixed_window(RateLimitItemPerMinute(3), key="same"),
            ]
        )

    assert storage.reset() == 0


def test_different_sliding_window_expiries_can_share_logical_key():
    storage = MemoryStorage(clock=FixedClock())
    limiter = CombinedRateLimiter(storage)

    result = limiter.hit_many(
        [
            Req.sliding_window_counter(RateLimitItemPerMinute(2), key="same"),
            Req.sliding_window_counter(RateLimitItemPerSecond(2), key="same"),
        ]
    )

    assert result


def test_mixed_fixed_and_sliding_windows_are_atomic():
    clock = FixedClock()
    storage = MemoryStorage(clock=clock)
    limiter = CombinedRateLimiter(storage)
    fixed_limit = RateLimitItemPerMinute(2)
    sliding_limit = RateLimitItemPerMinute(2)
    fill(storage, fixed_limit.key_for("fixed"), 60, 2, clock.value)

    fixed_request = fixed(fixed_limit, "fixed")
    sliding_request = sliding(sliding_limit, "sliding")
    result = limiter.hit_many([fixed_request, sliding_request])

    assert not result
    assert result.rejected == (fixed_request,)
    assert result.retry_at == clock.value + 60
    assert storage.get(fixed_limit.key_for("fixed")) == 2
    assert storage.get(sliding_limit.key_for("sliding")) == 0


def test_sliding_window_weighted_limit_failure_consumes_nothing():
    now = 1050.0
    storage = MemoryStorage(clock=FixedClock(now))
    limiter = CombinedRateLimiter(storage)
    limit = RateLimitItemPerMinute(2)
    previous_key = f"{limit.key_for('same')}/16"
    current_key = f"{limit.key_for('same')}/17"
    fill(storage, previous_key, 120, 2, 1020.0)
    fill(storage, current_key, 120, 2, 1020.0)

    result = limiter.hit_many([sliding(limit, "same")])

    assert not result
    assert result.rejected == (sliding(limit, "same"),)
    assert result.retry_at == 1080.0
    assert storage.storage[previous_key] == 2
    assert storage.storage[current_key] == 2

    storage.clock.value = 1079.9
    assert not limiter.hit_many([sliding(limit, "same")])
    storage.clock.value = 1080.000000001
    assert limiter.hit_many([sliding(limit, "same")])


def test_sliding_window_switch_uses_one_timestamp():
    clock = SteppedClock(119.9, 180.0)
    storage = MemoryStorage(clock=clock)
    limiter = CombinedRateLimiter(storage)
    limit = RateLimitItemPerMinute(1)
    previous_key = f"{limit.key_for('same')}/0"
    current_key = f"{limit.key_for('same')}/1"
    fill(storage, previous_key, 120, 1, clock.value)

    result = limiter.hit_many([sliding(limit, "same")])

    assert result.allowed
    assert storage.storage[previous_key] == 1
    assert storage.get(current_key) == 1


def test_fixed_window_switch_uses_one_timestamp():
    clock = SteppedClock(1060.0, 1120.0)
    storage = MemoryStorage(clock=clock)
    limiter = CombinedRateLimiter(storage)
    limit = RateLimitItemPerMinute(1)
    fill(storage, limit.key_for("same"), 60, 1, 1000.0)

    result = limiter.hit_many([fixed(limit, "same")])

    assert result
    assert storage.storage[limit.key_for("same")] == 1
    assert storage.get_expiry(limit.key_for("same")) == 1120.0


@pytest.mark.parametrize(
    "requests",
    [
        [],
        [object()],
        [
            Req(
                RateLimitItemPerMinute(1),
                (),
                1,
                "unknown-window",
            )
        ],
        [fixed(RateLimitItemPerMinute(1), "bad", cost=0)],
        [fixed(RateLimitItemPerMinute(1), "bad", cost=-1)],
    ],
)
def test_invalid_requests_do_not_change_state(requests):
    storage = MemoryStorage(clock=FixedClock())
    limiter = CombinedRateLimiter(storage)

    with pytest.raises((TypeError, ValueError)):
        limiter.hit_many(requests)

    assert storage.reset() == 0


def test_unsupported_backend_is_rejected_before_state_changes():

    class UnsupportedStorage(Storage):
        @property
        def base_exceptions(self):
            return ValueError

        def incr(self, key, expiry, amount=1):
            raise AssertionError

        def get(self, key):
            raise AssertionError

        def get_expiry(self, key):
            raise AssertionError

        def check(self):
            raise AssertionError

        def reset(self):
            raise AssertionError

        def clear(self, key):
            raise AssertionError

    with pytest.raises(NotImplementedError):
        CombinedRateLimiter(UnsupportedStorage())


def test_cost_over_limit_is_rejected_without_consumption():
    clock = FixedClock()
    storage = MemoryStorage(clock=clock)
    limiter = CombinedRateLimiter(storage)
    limit = RateLimitItemPerMinute(1)

    result = limiter.hit_many([fixed(limit, "same", cost=2)])

    assert not result
    assert result.retry_at == float("inf")
    assert storage.get(limit.key_for("same")) == 0


def test_reverse_order_concurrent_groups_do_not_deadlock():
    clock = FixedClock()
    storage = MemoryStorage(clock=clock)
    limiter = CombinedRateLimiter(storage)
    first = RateLimitItemPerMinute(100)
    second = RateLimitItemPerMinute(200)
    barrier = Barrier(2)

    def consume(keys):
        barrier.wait(timeout=5)
        return limiter.hit_many([fixed(first, keys[0]), fixed(second, keys[1])])

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(consume, [("a", "b"), ("b", "a")]))

    assert all(results)
    assert storage.storage[first.key_for("a")] == 1
    assert storage.storage[second.key_for("a")] == 1
    assert storage.storage[first.key_for("b")] == 1
    assert storage.storage[second.key_for("b")] == 1
