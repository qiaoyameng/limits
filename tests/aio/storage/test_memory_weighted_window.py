from __future__ import annotations

import asyncio
import random

import pytest

import limits.aio.storage.memory as memory_module
from limits.aio.storage import MemoryStorage
from tests.storage._weighted_window_reference import ReferenceMovingWindow


class Clock:
    def __init__(self) -> None:
        self.current = 0.0

    def time(self) -> float:
        return self.current


@pytest.fixture
async def storage(monkeypatch: pytest.MonkeyPatch) -> MemoryStorage:
    clock = Clock()
    monkeypatch.setattr(memory_module.time, "time", clock.time)
    storage = MemoryStorage()
    monkeypatch.setattr(
        storage,
        "_MemoryStorage__schedule_expiry",
        lambda: asyncio.sleep(0),
    )
    storage.clock = clock  # type: ignore[attr-defined]
    return storage


@pytest.mark.parametrize("seed", range(50))
async def test_weighted_window_matches_expanded_reference(
    storage: MemoryStorage, seed: int
) -> None:
    clock: Clock = storage.clock  # type: ignore[attr-defined]
    reference = ReferenceMovingWindow(clock)
    rng = random.Random(seed)
    limit = 10
    expiry = 5

    for index in range(60):
        clock.current += rng.choice([0.0, 0.0, 0.25, 1.0, 4.5, 6.0])
        cost = rng.choice([1, 1, 1, 2, 3, 4, limit + 1])
        await storage._MemoryStorage__expire_events()
        reference.expire()
        expected = reference.acquire(limit, expiry, cost)
        assert await storage.acquire_entry("key", limit, expiry, cost) is expected
        assert await storage.get_moving_window("key", limit, expiry) == (
            reference.get_moving_window(expiry)
        )
        if index % 7 == 0:
            await storage._MemoryStorage__expire_events()
            reference.expire()
            assert await storage.get_moving_window("key", limit, expiry) == (
                reference.get_moving_window(expiry)
            )

    await storage._MemoryStorage__expire_events()
    reference.expire()
    assert await storage.get_moving_window("key", limit, expiry) == (
        reference.get_moving_window(expiry)
    )
    assert sum(event.weight for event in storage.events.get("key", [])) == len(
        reference.events
    )
    assert len(storage.events.get("key", [])) <= len(reference.events)


async def test_large_weight_uses_one_event(storage: MemoryStorage) -> None:
    clock: Clock = storage.clock  # type: ignore[attr-defined]
    clock.current = 1
    assert await storage.acquire_entry("key", 1_000_000, 10, 1_000_000)
    assert len(storage.events["key"]) == 1
    assert storage.events["key"][0].weight == 1_000_000
    assert await storage.get_moving_window("key", 1_000_000, 10) == (
        1,
        1_000_000,
    )


async def test_few_large_weight_calls_do_not_expand_storage(
    storage: MemoryStorage,
) -> None:
    clock: Clock = storage.clock  # type: ignore[attr-defined]
    clock.current = 1
    assert await storage.acquire_entry("key", 1_000_000, 10)
    clock.current = 2
    assert await storage.acquire_entry("key", 1_000_000, 10, 999_999)

    assert len(storage.events["key"]) == 2
    assert sum(event.weight for event in storage.events["key"]) == 1_000_000
    assert await storage.get_moving_window("key", 1_000_000, 10) == (
        1,
        1_000_000,
    )

    clock.current = 3
    assert not await storage.acquire_entry("key", 1_000_000, 10)
    assert not await storage.acquire_entry("key", 1_000_000, 10, 1_000_001)
    assert len(storage.events["key"]) == 2
    assert sum(event.weight for event in storage.events["key"]) == 1_000_000


async def test_same_time_requests_merge_but_different_expiries_remain_separate(
    storage: MemoryStorage,
) -> None:
    clock: Clock = storage.clock  # type: ignore[attr-defined]
    clock.current = 1
    assert await storage.acquire_entry("key", 10, 5, 3)
    assert await storage.acquire_entry("key", 10, 10, 3)
    assert await storage.acquire_entry("key", 10, 5, 2)

    assert len(storage.events["key"]) == 2
    weights = sorted(event.weight for event in storage.events["key"])
    assert weights == [3, 5]

    clock.current = 7
    await storage._MemoryStorage__expire_events()
    assert len(storage.events["key"]) == 1
    assert storage.events["key"][0].weight == 3
    assert await storage.get_moving_window("key", 10, 10) == (1, 3)


async def test_event_order_is_preserved_after_short_expiry_cleanup(
    storage: MemoryStorage,
) -> None:
    clock: Clock = storage.clock  # type: ignore[attr-defined]
    clock.current = 1
    assert await storage.acquire_entry("key", 10, 2)
    clock.current = 2
    assert await storage.acquire_entry("key", 10, 10)
    clock.current = 4
    await storage._MemoryStorage__expire_events()
    assert await storage.acquire_entry("key", 10, 10)

    assert [event.atime for event in storage.events["key"]] == [4, 2]
    assert await storage.get_moving_window("key", 10, 10) == (2, 2)


async def test_rejection_leaves_no_partial_weight(storage: MemoryStorage) -> None:
    clock: Clock = storage.clock  # type: ignore[attr-defined]
    clock.current = 1
    assert await storage.acquire_entry("key", 10, 10, 9)
    assert not await storage.acquire_entry("key", 10, 10, 2)
    assert len(storage.events["key"]) == 1
    assert storage.events["key"][0].weight == 9
    assert await storage.get_moving_window("key", 10, 10) == (1, 9)


async def test_zero_weight_request_does_not_store_event(
    storage: MemoryStorage,
) -> None:
    clock: Clock = storage.clock  # type: ignore[attr-defined]
    clock.current = 1
    assert await storage.acquire_entry("key", 10, 10, 0)
    assert "key" not in storage.events


async def test_lock_cleanup_removes_expired_weighted_events(
    storage: MemoryStorage,
) -> None:
    clock: Clock = storage.clock  # type: ignore[attr-defined]
    clock.current = 1
    assert await storage.acquire_entry("key", 10, 5, 7)
    clock.current = 7
    await storage._MemoryStorage__expire_events()
    assert "key" not in storage.events
    assert await storage.get_moving_window("key", 10, 5) == (clock.time(), 0)


async def test_concurrent_weighted_deduction_is_atomic(
    storage: MemoryStorage,
) -> None:
    results = await asyncio.gather(
        *(storage.acquire_entry("key", 16, 60) for _ in range(20))
    )
    assert sum(results) == 16
    assert sum(event.weight for event in storage.events["key"]) == 16
    assert await storage.get_moving_window("key", 16, 60) == (0, 16)
