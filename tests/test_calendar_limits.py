from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone

import pytest

from limits import (
    CalendarPeriod,
    CalendarRateLimitItem,
    InvalidTimezoneError,
    RateLimitItemPerCalendarDay,
    RateLimitItemPerCalendarMonth,
    RateLimitItemPerCalendarWeek,
    RateLimitItemPerDay,
)
from limits.aio.storage import MemoryStorage as AsyncMemoryStorage
from limits.aio.strategies import (
    FixedWindowRateLimiter as AsyncFixedWindowRateLimiter,
)
from limits.storage import MemoryStorage
from limits.strategies import FixedWindowRateLimiter, MovingWindowRateLimiter


def utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> float:
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc).timestamp()


@dataclass(frozen=True)
class DayBoundaryCase:
    timezone: str
    before: float
    boundary: float
    before_window_hours: int
    after_window_hours: int
    repeated_time: float | None = None


DAY_BOUNDARY_CASES = [
    DayBoundaryCase(
        "America/New_York",
        utc(2030, 3, 10, 4, 59),
        utc(2030, 3, 10, 5),
        24,
        23,
    ),
    DayBoundaryCase(
        "America/New_York",
        utc(2030, 11, 3, 3, 59),
        utc(2030, 11, 3, 4),
        24,
        25,
        utc(2030, 11, 3, 6, 30),
    ),
    DayBoundaryCase(
        "Pacific/Auckland",
        utc(2030, 9, 28, 11, 59),
        utc(2030, 9, 28, 12),
        24,
        23,
    ),
    DayBoundaryCase(
        "Pacific/Auckland",
        utc(2030, 4, 6, 10, 59),
        utc(2030, 4, 6, 11),
        24,
        25,
    ),
]


@pytest.fixture
def freeze(monkeypatch: pytest.MonkeyPatch):
    import limits.aio.storage.memory as async_memory
    import limits.aio.strategies as async_strategies
    import limits.storage.memory as sync_memory
    import limits.strategies as sync_strategies
    from limits import calendar

    def freeze_at(at: float) -> None:
        monkeypatch.setattr(calendar.time, "time", lambda: at)
        monkeypatch.setattr(sync_strategies.time, "time", lambda: at)
        monkeypatch.setattr(sync_memory.time, "time", lambda: at)
        monkeypatch.setattr(async_strategies.time, "time", lambda: at)
        monkeypatch.setattr(async_memory.time, "time", lambda: at)

    return freeze_at


class TestCalendarFixedWindow:
    @pytest.mark.parametrize("case", DAY_BOUNDARY_CASES)
    def test_daily_boundary(
        self,
        freeze,
        case: DayBoundaryCase,
    ) -> None:
        storage = MemoryStorage()
        limiter = FixedWindowRateLimiter(storage)
        limit = RateLimitItemPerCalendarDay(2, case.timezone)
        identifier = f"daily-{case.timezone}"

        before_window = limit.get_window(case.before)
        assert (
            before_window.end - before_window.start
            == case.before_window_hours * 60 * 60
        )

        freeze(case.before)
        before_key = limit.key_for(identifier, at=case.before)
        assert limiter.hit(limit, identifier)
        assert limiter.hit(limit, identifier)
        assert storage.expirations[before_key] == before_window.end
        assert not limiter.test(limit, identifier)
        stats = limiter.get_window_stats(limit, identifier)
        assert stats.remaining == 0
        assert stats.reset_time == before_window.end

        freeze(case.boundary)
        after_window = limit.get_window(case.boundary)
        assert (
            after_window.end - after_window.start
            == case.after_window_hours * 60 * 60
        )
        after_key = limit.key_for(identifier, at=case.boundary)
        assert before_key != after_key
        assert limiter.test(limit, identifier)
        assert limiter.hit(limit, identifier)
        assert limiter.hit(limit, identifier)
        stats = limiter.get_window_stats(limit, identifier)
        assert stats.remaining == 0
        assert stats.reset_time == after_window.end

        if case.repeated_time is not None:
            freeze(case.repeated_time)
            assert limit.key_for(identifier, at=case.repeated_time) == after_key
            assert not limiter.test(limit, identifier)
            assert (
                limiter.get_window_stats(limit, identifier).reset_time
                == after_window.end
            )

    def test_week_uses_selected_start_and_exact_cross_year_identity(
        self, freeze
    ) -> None:
        storage = MemoryStorage()
        limiter = FixedWindowRateLimiter(storage)
        limit = RateLimitItemPerCalendarWeek(1, "UTC", week_start=6)

        before = utc(2030, 1, 5, 23)
        boundary = utc(2030, 1, 6)

        freeze(before)
        before_key = limit.key_for("user", at=before)
        assert "/calendar/week/UTC/2029-W52-w6" in before_key
        assert limiter.hit(limit, "user")
        assert limiter.get_window_stats(limit, "user").reset_time == boundary

        freeze(boundary)
        assert limit.key_for("user", at=boundary) != before_key
        assert "/calendar/week/UTC/2030-W01-w6" in limit.key_for(
            "user", at=boundary
        )
        assert limiter.hit(limit, "user")
        assert (
            limiter.get_window_stats(limit, "user").reset_time
            == utc(2030, 1, 13)
        )

    def test_month_handles_leap_year_and_dst_months(self) -> None:
        february = RateLimitItemPerCalendarMonth(1, "UTC")
        feb_window = february.get_window(utc(2032, 2, 29, 23))
        assert feb_window.start == utc(2032, 2, 1)
        assert feb_window.end == utc(2032, 3, 1)
        assert feb_window.end - feb_window.start == 29 * 24 * 60 * 60

        march = RateLimitItemPerCalendarMonth(1, "America/New_York")
        march_window = march.get_window(utc(2030, 3, 15))
        assert march_window.start == utc(2030, 3, 1, 5)
        assert march_window.end == utc(2030, 4, 1, 4)
        assert march_window.end - march_window.start == 743 * 60 * 60

        november = RateLimitItemPerCalendarMonth(1, "America/New_York")
        november_window = november.get_window(utc(2030, 11, 15))
        assert november_window.start == utc(2030, 11, 1, 4)
        assert november_window.end == utc(2030, 12, 1, 5)
        assert november_window.end - november_window.start == 721 * 60 * 60

    def test_nonexistent_midnight_uses_transition_boundary(self) -> None:
        limit = RateLimitItemPerCalendarDay(1, "America/Havana")
        window = limit.get_window(utc(2023, 3, 11, 5, 30))

        assert window.start == utc(2023, 3, 11, 5)
        assert window.end == utc(2023, 3, 12, 5)
        assert window.end - window.start == 24 * 60 * 60

    def test_invalid_configuration_fails_before_first_deduction(self, freeze) -> None:
        freeze(utc(2030, 1, 1))
        storage = MemoryStorage()

        with pytest.raises(InvalidTimezoneError):
            RateLimitItemPerCalendarDay(1, "Not/A/Zone")
        with pytest.raises(ValueError, match="period"):
            CalendarRateLimitItem(1, "year", "UTC")
        with pytest.raises(ValueError, match="week_start"):
            RateLimitItemPerCalendarWeek(1, "UTC", week_start=7)

        assert storage.get("anything") == 0

    def test_existing_fixed_second_granularity_is_unchanged(self) -> None:
        limit = RateLimitItemPerDay(1)
        assert limit.get_expiry() == 24 * 60 * 60
        assert limit.key_for().endswith("/1/1/day")

    def test_calendar_limits_are_not_equal_to_fixed_seconds_limits(self) -> None:
        fixed_limit = RateLimitItemPerDay(1)
        calendar_limit = RateLimitItemPerCalendarDay(1, "UTC")

        assert fixed_limit != calendar_limit
        assert calendar_limit != fixed_limit
        assert calendar_limit != RateLimitItemPerCalendarDay(
            1, "UTC", namespace="OTHER"
        )

    def test_moving_window_rejects_calendar_limits(self) -> None:
        storage = MemoryStorage()
        limiter = MovingWindowRateLimiter(storage)
        limit = RateLimitItemPerCalendarDay(1, "UTC")

        with pytest.raises(NotImplementedError):
            limiter.hit(limit)
        with pytest.raises(NotImplementedError):
            limiter.clear(limit)


class TestAsyncCalendarFixedWindow:
    @pytest.mark.parametrize("case", DAY_BOUNDARY_CASES)
    def test_daily_boundary(self, freeze, case: DayBoundaryCase) -> None:
        asyncio.run(self._test_daily_boundary(freeze, case))

    async def _test_daily_boundary(
        self, freeze, case: DayBoundaryCase
    ) -> None:
        storage = AsyncMemoryStorage()
        limiter = AsyncFixedWindowRateLimiter(storage)
        limit = RateLimitItemPerCalendarDay(2, case.timezone)
        identifier = f"async-daily-{case.timezone}"

        freeze(case.before)
        before_key = limit.key_for(identifier, at=case.before)
        assert await limiter.hit(limit, identifier)
        assert await limiter.hit(limit, identifier)
        assert storage.expirations[before_key] == limit.get_reset_at(case.before)
        assert not await limiter.test(limit, identifier)
        stats = await limiter.get_window_stats(limit, identifier)
        assert stats.remaining == 0
        assert stats.reset_time == limit.get_reset_at(case.before)

        freeze(case.boundary)
        assert limit.key_for(identifier, at=case.boundary) != before_key
        assert await limiter.hit(limit, identifier)
        stats = await limiter.get_window_stats(limit, identifier)
        assert stats.remaining == 1
        assert stats.reset_time == limit.get_reset_at(case.boundary)

    def test_invalid_configuration_fails_before_first_deduction(self, freeze) -> None:
        asyncio.run(self._test_invalid_configuration(freeze))

    async def _test_invalid_configuration(self, freeze) -> None:
        freeze(utc(2030, 1, 1))
        storage = AsyncMemoryStorage()

        with pytest.raises(InvalidTimezoneError):
            CalendarRateLimitItem(1, CalendarPeriod.DAY, "Not/A/Zone")
        with pytest.raises(ValueError, match="period"):
            CalendarRateLimitItem(1, "year", "UTC")
        with pytest.raises(ValueError, match="week_start"):
            CalendarRateLimitItem(1, "week", "UTC", week_start=-1)

        assert await storage.get("anything") == 0

    def test_calendar_window_accepts_aware_utc_datetime(self) -> None:
        asyncio.run(self._test_calendar_window_accepts_aware_utc_datetime())

    async def _test_calendar_window_accepts_aware_utc_datetime(self) -> None:
        limit = RateLimitItemPerCalendarDay(1, "UTC")
        moment = datetime(2030, 1, 1, 12, tzinfo=timezone.utc)

        assert limit.get_window(moment) == limit.get_window(moment.timestamp())
        with pytest.raises(ValueError):
            limit.get_window(moment.replace(tzinfo=None))
