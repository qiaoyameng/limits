from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from limits import (
    CalendarPeriod,
    CalendarRateLimitItem,
    CalendarRateLimitItemPerDay,
    CalendarRateLimitItemPerMonth,
    CalendarRateLimitItemPerWeek,
)
from limits.aio.storage import MemoryStorage as AsyncMemoryStorage
from limits.aio.strategies import (
    CalendarFixedWindowRateLimiter as AsyncCalendarFixedWindowRateLimiter,
)
from limits.limits import RateLimitItemPerDay
from limits.storage import MemoryStorage
from limits.strategies import CalendarFixedWindowRateLimiter


def utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> float:
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc).timestamp()


def assert_window_resets(limiter, item, reset_at: float) -> None:
    before = reset_at - 1
    assert item.reset_at(before) == reset_at
    assert item.get_expiry(before) == pytest.approx(1.0)

    old_key = item.key_for("customer", at=before)
    assert limiter.hit(item, "customer", at=before)
    assert limiter.test(item, "customer", at=before)
    assert not limiter.test(item, "customer", at=before, cost=2)
    assert not limiter.hit(item, "customer", at=before, cost=2)

    stats = limiter.get_window_stats(item, "customer", at=before)
    assert stats.remaining == 0
    assert stats.reset_time == reset_at

    assert not limiter.hit(item, "customer", at=before)
    assert limiter.hit(item, "customer", at=reset_at)
    assert item.key_for("customer", at=reset_at) != old_key


async def assert_async_window_resets(limiter, item, reset_at: float) -> None:
    before = reset_at - 1
    assert item.reset_at(before) == reset_at
    old_key = item.key_for("customer", at=before)

    assert await limiter.hit(item, "customer", at=before)
    assert await limiter.test(item, "customer", at=before)
    assert not await limiter.test(item, "customer", at=before, cost=2)
    assert not await limiter.hit(item, "customer", at=before, cost=2)

    stats = await limiter.get_window_stats(item, "customer", at=before)
    assert stats.remaining == 0
    assert stats.reset_time == reset_at

    assert not await limiter.hit(item, "customer", at=before)
    assert await limiter.hit(item, "customer", at=reset_at)
    assert item.key_for("customer", at=reset_at) != old_key


def test_calendar_day_dst_boundaries_sync_memory():
    limiter = CalendarFixedWindowRateLimiter(MemoryStorage())
    cases = [
        ("America/New_York", utc(2026, 3, 9, 4)),
        ("America/New_York", utc(2026, 11, 2, 5)),
        ("Pacific/Auckland", utc(2026, 9, 27, 11)),
        ("Pacific/Auckland", utc(2026, 4, 5, 12)),
        ("America/Havana", utc(2026, 3, 8, 5)),
        ("America/Havana", utc(2026, 11, 2, 5)),
    ]

    for zone, reset_at in cases:
        item = CalendarRateLimitItem(2, CalendarPeriod.DAY, zone)
        assert_window_resets(limiter, item, reset_at)


async def run_async_dst_boundaries():
    limiter = AsyncCalendarFixedWindowRateLimiter(AsyncMemoryStorage())
    try:
        for zone, reset_at in [
            ("America/New_York", utc(2026, 3, 9, 4)),
            ("America/New_York", utc(2026, 11, 2, 5)),
            ("Pacific/Auckland", utc(2026, 9, 27, 11)),
            ("Pacific/Auckland", utc(2026, 4, 5, 12)),
        ]:
            item = CalendarRateLimitItem(2, CalendarPeriod.DAY, zone)
            await assert_async_window_resets(limiter, item, reset_at)
    finally:
        await limiter.storage.reset()


def test_calendar_day_dst_boundaries_async_memory():
    asyncio.run(run_async_dst_boundaries())


def test_calendar_week_uses_local_start_and_exact_identity():
    limiter = CalendarFixedWindowRateLimiter(MemoryStorage())
    item = CalendarRateLimitItem(2, CalendarPeriod.WEEK, "Pacific/Auckland")
    reset_at = utc(2026, 1, 4, 11)

    assert_window_resets(limiter, item, reset_at)
    key = item.key_for("customer", at=reset_at - 1)
    assert "/calendar/Pacific/Auckland/week/0/2025-12-29" in key

    sunday = CalendarRateLimitItem(
        1, CalendarPeriod.WEEK, "America/New_York", week_start="sunday"
    )
    sunday_key = sunday.key_for(at=utc(2026, 1, 1))
    assert "/calendar/America/New_York/week/6/2025-12-28" in sunday_key


def test_calendar_month_handles_month_end_and_leap_day():
    limiter = CalendarFixedWindowRateLimiter(MemoryStorage())

    leap_february = CalendarRateLimitItem(2, CalendarPeriod.MONTH, "UTC")
    assert_window_resets(limiter, leap_february, utc(2024, 3, 1))

    auckland_april = CalendarRateLimitItem(
        2, CalendarPeriod.MONTH, "Pacific/Auckland"
    )
    assert_window_resets(limiter, auckland_april, utc(2026, 4, 30, 12))


def test_empty_window_reports_next_boundary_and_remaining_amount():
    limiter = CalendarFixedWindowRateLimiter(MemoryStorage())
    item = CalendarRateLimitItem(3, CalendarPeriod.DAY, "America/New_York")
    at = utc(2026, 3, 9, 3, 59)

    stats = limiter.get_window_stats(item, at=at)
    assert stats.remaining == 3
    assert stats.reset_time == utc(2026, 3, 9, 4)


def test_calendar_window_accepts_timezone_aware_datetime():
    limiter = CalendarFixedWindowRateLimiter(MemoryStorage())
    item = CalendarRateLimitItem(1, CalendarPeriod.DAY, "UTC")
    moment = datetime(2024, 2, 29, 23, 59, tzinfo=timezone.utc)

    assert limiter.hit(item, at=moment)
    assert not limiter.test(item, at=moment)
    stats = limiter.get_window_stats(item, at=moment)
    assert stats.reset_time == utc(2024, 3, 1)

    with pytest.raises(ValueError):
        item.get_window(datetime(2024, 2, 29, 23, 59))


@pytest.mark.parametrize(
    ("period", "timezone", "kwargs"),
    [
        ("year", "UTC", {}),
        ("day", "Mars/Phobos", {}),
        ("week", "UTC", {"week_start": 7}),
        ("week", "UTC", {"week_start": "funday"}),
    ],
)
def test_invalid_calendar_configuration_raises_before_hit(period, timezone, kwargs):
    with pytest.raises(ValueError):
        CalendarRateLimitItem(1, period, timezone, **kwargs)


@pytest.mark.parametrize(
    ("amount", "multiples"),
    [(0, 1), (1, 2), (-1, 1)],
)
def test_invalid_amount_and_multiples(amount, multiples):
    with pytest.raises(ValueError):
        CalendarRateLimitItem(
            amount, CalendarPeriod.DAY, "UTC", multiples=multiples
        )


def test_fixed_second_day_granularity_is_unchanged():
    assert RateLimitItemPerDay(1).get_expiry() == 24 * 60 * 60
    item = RateLimitItemPerDay(1)
    assert item.key_for() == "LIMITER/1/1/day"


def test_calendar_convenience_classes_match_general_rule():
    assert CalendarRateLimitItemPerDay(1, "UTC") == CalendarRateLimitItem(
        1, CalendarPeriod.DAY, "UTC"
    )
    assert CalendarRateLimitItemPerWeek(
        1, "UTC", week_start="sunday"
    ) == CalendarRateLimitItem(1, CalendarPeriod.WEEK, "UTC", week_start="sunday")
    assert CalendarRateLimitItemPerMonth(1, "UTC") == CalendarRateLimitItem(
        1, CalendarPeriod.MONTH, "UTC"
    )
