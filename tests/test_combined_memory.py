from __future__ import annotations

import threading

import pytest

from limits import CombinedRateLimitRequest, RateLimitItemPerSecond
from limits.storage import MemoryStorage
from limits.storage.base import Storage
from limits.storage.memory import CombinedWindowRequest
from limits.strategies import CombinedRateLimiter


class Clock:
    def __init__(self, now: float = 100.0) -> None:
        self.now = now

    def time(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    clock = Clock()
    monkeypatch.setattr("limits.storage.memory.time.time", clock.time)
    return clock


def test_failure_does_not_consume(clock: Clock) -> None:
    storage = MemoryStorage()
    limiter = CombinedRateLimiter(storage)
    user = RateLimitItemPerSecond(1)
    tenant = RateLimitItemPerSecond(2)
    requests = [
        CombinedRateLimitRequest.fixed(user, "user", cost=1, source="user"),
        CombinedRateLimitRequest.fixed(tenant, "tenant", cost=2, source="tenant"),
    ]

    assert limiter.hit_many(requests).allowed
    result = limiter.hit_many(requests)

    assert not result.allowed
    assert [rejection.source for rejection in result.rejected] == ["user", "tenant"]
    assert result.retry_at == 101.0
    assert storage.get(user.key_for("user")) == 1
    assert storage.get(tenant.key_for("tenant")) == 2


def test_repeated_storage_key_merges_demand(clock: Clock) -> None:
    storage = MemoryStorage()
    limiter = CombinedRateLimiter(storage)
    limit = RateLimitItemPerSecond(2)

    result = limiter.hit_many(
        [
            CombinedRateLimitRequest.fixed(limit, "shared", source=0),
            CombinedRateLimitRequest.fixed(limit, "shared", source=1),
        ]
    )

    assert result.allowed
    assert storage.get(limit.key_for("shared")) == 2
    assert limiter.hit_many(
        [
            CombinedRateLimitRequest.fixed(limit, "shared", source=2),
            CombinedRateLimitRequest.fixed(limit, "shared", source=3),
        ]
    ).rejected[0].source == 2
    assert storage.get(limit.key_for("shared")) == 2


def test_distinct_rules_using_same_storage_key_are_rejected(clock: Clock) -> None:
    storage = MemoryStorage()
    with pytest.raises(ValueError):
        storage.acquire_combined_entries(
            [
                CombinedWindowRequest("shared/100", 1, 10),
                CombinedWindowRequest(
                    "shared",
                    1,
                    10,
                    strategy="sliding-window-counter"),
            ]
        )
    assert storage.storage.get("shared/100") is None


def test_combined_request_uses_one_decision_time(
    clock: Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = MemoryStorage()
    limit = RateLimitItemPerSecond(1)
    calls = 0

    def advancing_time() -> float:
        nonlocal calls
        calls += 1
        return 100.999 + (calls - 1) * 0.001

    monkeypatch.setattr("limits.storage.memory.time.time", advancing_time)
    storage.storage[limit.key_for("first")] = 1
    storage.storage[limit.key_for("second")] = 1
    storage.expirations[limit.key_for("first")] = 101.0
    storage.expirations[limit.key_for("second")] = 101.0

    result = storage.acquire_combined_entries(
        [
            CombinedWindowRequest(limit.key_for("first"), 1, 1),
            CombinedWindowRequest(limit.key_for("second"), 1, 1),
        ]
    )

    assert not result.allowed
    assert result.retry_at == 101.0
    assert storage.storage[limit.key_for("first")] == 1
    assert storage.storage[limit.key_for("second")] == 1


def test_combined_fixed_window_switch(clock: Clock) -> None:
    storage = MemoryStorage()
    limit = RateLimitItemPerSecond(1)
    key = limit.key_for("boundary")
    storage.incr(key, 1)
    assert storage.get(key) == 1

    clock.now = 101.0
    result = storage.acquire_combined_entries([CombinedWindowRequest(key, 1, 1)])

    assert result.allowed
    assert storage.get(key) == 1
    assert storage.get_expiry(key) == 102.0


def test_sliding_window_failure_does_not_consume(clock: Clock) -> None:
    storage = MemoryStorage()
    limiter = CombinedRateLimiter(storage)
    limit = RateLimitItemPerSecond(2)
    requests = [
        CombinedRateLimitRequest.sliding(limit, "user"),
        CombinedRateLimitRequest.sliding(limit, "tenant"),
    ]

    assert limiter.hit_many(requests).allowed
    result = limiter.hit_many(
        [
            CombinedRateLimitRequest.sliding(limit, "user"),
            CombinedRateLimitRequest.sliding(limit, "tenant", cost=2),
        ]
    )

    assert not result.allowed
    assert storage.get(f"{limit.key_for('user')}/100") == 1
    assert storage.get(f"{limit.key_for('tenant')}/100") == 1


def test_sliding_window_switch_uses_new_window(clock: Clock) -> None:
    storage = MemoryStorage()
    limit = RateLimitItemPerSecond(1)
    key = limit.key_for("boundary")

    assert storage.acquire_combined_entries(
        [CombinedWindowRequest(key, 1, 1, strategy="sliding-window-counter")]
    ).allowed
    clock.now = 101.0
    result = storage.acquire_combined_entries(
        [CombinedWindowRequest(key, 1, 1, strategy="sliding-window-counter")]
    )

    assert result.allowed
    assert storage.get(f"{key}/101") == 1


def test_empty_group_and_invalid_cost_rejected_before_state(clock: Clock) -> None:
    storage = MemoryStorage()
    with pytest.raises(ValueError):
        storage.acquire_combined_entries([])
    with pytest.raises(ValueError):
        storage.acquire_combined_entries([CombinedWindowRequest("key", 1, 1, cost=0)])
    assert storage.storage == {}


def test_unsupported_backend_rejected(clock: Clock) -> None:
    unsupported_storage = type(
        "UnsupportedStorage",
        (Storage,),
        {
            "incr": lambda self, key, expiry, amount=1: 0,
            "get": lambda self, key: 0,
            "get_expiry": lambda self, key: 0.0,
            "check": lambda self: True,
            "reset": lambda self: 0,
            "clear": lambda self, key: None,
            "base_exceptions": property(lambda self: ValueError),
        },
    )()

    with pytest.raises(NotImplementedError):
        CombinedRateLimiter(unsupported_storage)


def test_reverse_lock_order_concurrency(clock: Clock) -> None:
    storage = MemoryStorage()
    results: dict[str, bool] = {}

    def run(name: str, keys: tuple[str, str]) -> None:
        results[name] = storage.acquire_combined_entries(
            [
                CombinedWindowRequest(keys[0], 1, 1),
                CombinedWindowRequest(keys[1], 1, 1),
            ]
        ).allowed

    first = threading.Thread(target=run, args=("first", ("a", "b")))
    second = threading.Thread(target=run, args=("second", ("b", "a")))
    first.start()
    second.start()
    first.join(1)
    second.join(1)

    assert not first.is_alive()
    assert not second.is_alive()
    assert sum(results.values()) == 1
    assert storage.get("a") == 1
    assert storage.get("b") == 1
