from __future__ import annotations

import asyncio

import pytest

from limits import CombinedRateLimitRequest, RateLimitItemPerSecond
from limits.aio.storage import MemoryStorage
from limits.aio.strategies import CombinedRateLimiter
from limits.storage.memory import CombinedWindowRequest


class Clock:
    def __init__(self, now: float = 100.0) -> None:
        self.now = now

    def time(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    clock = Clock()
    monkeypatch.setattr("limits.aio.storage.memory.time.time", clock.time)
    monkeypatch.setattr("limits.storage.memory.time.time", clock.time)
    return clock


async def test_async_failure_does_not_consume(clock: Clock) -> None:
    storage = MemoryStorage()
    limiter = CombinedRateLimiter(storage)
    user = RateLimitItemPerSecond(1)
    tenant = RateLimitItemPerSecond(2)
    requests = [
        CombinedRateLimitRequest.fixed(user, "user", source="user"),
        CombinedRateLimitRequest.fixed(tenant, "tenant", cost=2, source="tenant"),
    ]

    assert (await limiter.hit_many(requests)).allowed
    result = await limiter.hit_many(requests)

    assert not result
    assert {rejection.source for rejection in result.rejected} == {"user", "tenant"}
    assert await storage.get(user.key_for("user")) == 1
    assert await storage.get(tenant.key_for("tenant")) == 2


async def test_async_repeated_storage_key_merges_demand(clock: Clock) -> None:
    storage = MemoryStorage()
    limiter = CombinedRateLimiter(storage)
    limit = RateLimitItemPerSecond(2)

    result = await limiter.hit_many(
        [
            CombinedRateLimitRequest.fixed(limit, "shared"),
            CombinedRateLimitRequest.fixed(limit, "shared"),
        ]
    )
    second = await limiter.hit_many(
        [
            CombinedRateLimitRequest.fixed(limit, "shared"),
            CombinedRateLimitRequest.fixed(limit, "shared"),
        ]
    )

    assert result.allowed
    assert not second.allowed
    assert await storage.get(limit.key_for("shared")) == 2


async def test_async_fixed_window_switch_and_validation(clock: Clock) -> None:
    storage = MemoryStorage()
    limit = RateLimitItemPerSecond(1)
    key = limit.key_for("boundary")
    await storage.incr(key, 1)

    clock.now = 101.0
    result = await storage.acquire_combined_entries(
        [CombinedWindowRequest(key, 1, 1)]
    )

    assert result.allowed
    assert await storage.get(key) == 1

    with pytest.raises(ValueError):
        await storage.acquire_combined_entries([])
    with pytest.raises(ValueError):
        await storage.acquire_combined_entries(
            [CombinedWindowRequest(key, 1, 1, cost=0)]
        )


async def test_async_sliding_window_switch(clock: Clock) -> None:
    storage = MemoryStorage()
    limit = RateLimitItemPerSecond(1)
    key = limit.key_for("boundary")
    request = CombinedWindowRequest(
        key, 1, 1, strategy="sliding-window-counter"
    )

    assert (await storage.acquire_combined_entries([request])).allowed
    clock.now = 101.0
    assert (await storage.acquire_combined_entries([request])).allowed
    assert await storage.get(f"{key}/101") == 1


async def test_async_reverse_lock_order_concurrency(clock: Clock) -> None:
    storage = MemoryStorage()
    results: list[bool] = []

    async def run(keys: tuple[str, str]) -> None:
        result = await storage.acquire_combined_entries(
            [
                CombinedWindowRequest(keys[0], 1, 1),
                CombinedWindowRequest(keys[1], 1, 1),
            ]
        )
        results.append(result.allowed)

    first = asyncio.create_task(run(("a", "b")))
    second = asyncio.create_task(run(("b", "a")))
    await asyncio.wait_for(asyncio.gather(first, second), timeout=1)

    assert sum(results) == 1
    assert storage.storage["a"] == 1
    assert storage.storage["b"] == 1
