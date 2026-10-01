from __future__ import annotations

import asyncio

import pytest

from limits.aio.storage import MemoryStorage
from limits.aio.strategies import CombinedRateLimiter
from limits.combined import CombinedRateLimitRequest as Req
from limits.limits import RateLimitItemPerMinute


class FixedClock:
    def __init__(self, value=1000.0):
        self.value = value

    def __call__(self):
        return self.value


class CountingClock(FixedClock):
    def __init__(self):
        super().__init__(1000.0)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.value


def fixed(item, *identifiers, cost=1):
    return Req.fixed_window(item, *identifiers, cost=cost)


def sliding(item, *identifiers, cost=1):
    return Req.sliding_window_counter(item, *identifiers, cost=cost)


async def _failure_consumes_nothing():
    clock = FixedClock()
    storage = MemoryStorage(clock=clock)
    limiter = CombinedRateLimiter(storage)
    user = RateLimitItemPerMinute(2)
    tenant = RateLimitItemPerMinute(1)
    tenant_request = fixed(tenant, "tenant")
    storage.storage[user.key_for("user")] = 1
    storage.expirations[user.key_for("user")] = clock.value + 60
    storage.storage[tenant.key_for("tenant")] = 1
    storage.expirations[tenant.key_for("tenant")] = clock.value + 60

    result = await limiter.hit_many([fixed(user, "user"), tenant_request])

    assert not result
    assert result.rejected == (tenant_request,)
    assert result.retry_at == clock.value + 60
    assert storage.storage[user.key_for("user")] == 1
    assert storage.storage[tenant.key_for("tenant")] == 1


def test_async_group_failure_consumes_nothing():
    asyncio.run(_failure_consumes_nothing())


async def _duplicate_keys_merge_cost():
    storage = MemoryStorage(clock=FixedClock())
    limiter = CombinedRateLimiter(storage)
    limit = RateLimitItemPerMinute(2)
    requests = [fixed(limit, "same"), fixed(limit, "same")]

    assert await limiter.hit_many(requests)
    assert storage.storage[limit.key_for("same")] == 2
    assert not await limiter.hit_many(requests)
    assert storage.storage[limit.key_for("same")] == 2


def test_async_duplicate_keys_merge_cost():
    asyncio.run(_duplicate_keys_merge_cost())


async def _mixed_strategies_use_one_judgement_timestamp():
    clock = CountingClock()
    storage = MemoryStorage(clock=clock)
    limiter = CombinedRateLimiter(storage)
    fixed_limit = RateLimitItemPerMinute(1)
    sliding_limit = RateLimitItemPerMinute(1)

    result = await limiter.hit_many(
        [fixed(fixed_limit, "fixed"), sliding(sliding_limit, "sliding")]
    )

    assert result
    assert clock.calls == 1


def test_async_mixed_strategies_use_one_judgement_timestamp():
    asyncio.run(_mixed_strategies_use_one_judgement_timestamp())


async def _invalid_empty_group_does_not_change_state():
    storage = MemoryStorage(clock=FixedClock())
    limiter = CombinedRateLimiter(storage)

    with pytest.raises(ValueError):
        await limiter.hit_many([])

    assert await storage.reset() == 0


def test_async_invalid_empty_group_does_not_change_state():
    asyncio.run(_invalid_empty_group_does_not_change_state())


async def _reverse_order_concurrent_groups_do_not_deadlock():
    storage = MemoryStorage(clock=FixedClock())
    limiter = CombinedRateLimiter(storage)
    first = RateLimitItemPerMinute(100)
    second = RateLimitItemPerMinute(200)
    started = asyncio.Event()
    release = asyncio.Event()

    async def consume(keys, should_start):
        if should_start:
            started.set()
        else:
            await started.wait()
            await asyncio.sleep(0)
        await release.wait()
        return await limiter.hit_many(
            [fixed(first, keys[0]), fixed(second, keys[1])]
        )

    task_a = asyncio.create_task(consume(("a", "b"), True))
    task_b = asyncio.create_task(consume(("b", "a"), False))
    await asyncio.sleep(0)
    release.set()
    results = await asyncio.gather(task_a, task_b)

    assert all(results)
    assert storage.storage[first.key_for("b")] == 1
    assert storage.storage[second.key_for("b")] == 1
    assert storage.storage[first.key_for("a")] == 1
    assert storage.storage[second.key_for("a")] == 1


def test_async_reverse_order_concurrent_groups_do_not_deadlock():
    asyncio.run(_reverse_order_concurrent_groups_do_not_deadlock())
