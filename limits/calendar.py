from __future__ import annotations

import time
from datetime import date, datetime, timedelta
from enum import Enum
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from limits.typing import NamedTuple

from .limits import Granularity, RateLimitItem, safe_string


class CalendarPeriod(str, Enum):
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


class CalendarWindow(NamedTuple):
    start: float
    reset_at: float
    key: str


_WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


def _period_value(period: CalendarPeriod | str) -> str:
    if isinstance(period, CalendarPeriod):
        return period.value
    if isinstance(period, str):
        value = period.lower()
        if value in {period.value for period in CalendarPeriod}:
            return value
    raise ValueError(f"unsupported calendar period: {period!r}")


def _week_start_value(week_start: int | str | None) -> int:
    if week_start is None:
        return 0
    if isinstance(week_start, bool) or not isinstance(week_start, int):
        if isinstance(week_start, str):
            try:
                return _WEEKDAYS[week_start.lower()]
            except KeyError:
                pass
        raise ValueError(f"unsupported week start day: {week_start!r}")
    if not 0 <= week_start <= 6:
        raise ValueError("week start day must be between 0 (Monday) and 6 (Sunday)")
    return week_start


def _zoneinfo(value: str | ZoneInfo) -> tuple[ZoneInfo, str]:
    if isinstance(value, ZoneInfo):
        return value, value.key or str(value)
    if not isinstance(value, str):
        raise ValueError("timezone must be an IANA timezone name")
    try:
        timezone = ZoneInfo(value)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(f"unknown IANA timezone: {value!r}") from exc
    return timezone, value


def _timestamp(at: float | datetime | None) -> float | None:
    if at is None or isinstance(at, (int, float)):
        return at
    if isinstance(at, datetime):
        if at.tzinfo is None:
            raise ValueError("at must be a timestamp or timezone-aware datetime")
        return at.timestamp()
    raise TypeError("at must be a timestamp or timezone-aware datetime")


class CalendarRateLimitItem(RateLimitItem):
    """
    A fixed calendar window limited by a day, week, or month boundary in an
    IANA timezone.

    Week start days are numbered 0 through 6, where Monday is 0 and Sunday is
    6. Weekday names are also accepted.
    """

    __slots__ = ["period", "timezone", "week_start", "_tzinfo"]

    def __init__(
        self,
        amount: int,
        period: CalendarPeriod | str,
        timezone: str | ZoneInfo,
        multiples: int | None = 1,
        week_start: int | str | None = None,
        namespace: str = "LIMITER",
    ) -> None:
        if multiples is not None and int(multiples) != 1:
            raise ValueError("calendar rate limits do not support multiples")
        super().__init__(amount, 1, namespace)
        if self.amount < 1:
            raise ValueError("amount must be a positive integer")

        self.period = _period_value(period)
        self._tzinfo, self.timezone = _zoneinfo(timezone)
        resolved_week_start = _week_start_value(week_start)
        self.week_start = (
            resolved_week_start
            if self.period == CalendarPeriod.WEEK.value
            else 0
        )

    def _start_date(self, local_date: date) -> date:
        if self.period == CalendarPeriod.DAY.value:
            return local_date
        if self.period == CalendarPeriod.WEEK.value:
            return local_date - timedelta(
                days=(local_date.weekday() - self.week_start) % 7
            )
        return local_date.replace(day=1)

    def _next_start_date(self, start_date: date) -> date:
        if self.period == CalendarPeriod.MONTH.value:
            return (start_date.replace(day=1) + timedelta(days=32)).replace(day=1)
        days = 7 if self.period == CalendarPeriod.WEEK.value else 1
        return start_date + timedelta(days=days)

    def _midnight(self, local_date: date) -> float:
        return datetime(
            local_date.year,
            local_date.month,
            local_date.day,
            tzinfo=self._tzinfo,
        ).timestamp()

    def get_window(self, at: float | datetime | None = None) -> CalendarWindow:
        """
        Return the UTC start, reset time, and exact identity of the calendar
        window containing ``at``.
        """
        at = _timestamp(at)
        if at is None:
            at = time.time()
        local_date = datetime.fromtimestamp(at, self._tzinfo).date()
        start_date = self._start_date(local_date)
        end_date = self._next_start_date(start_date)

        if self.period == CalendarPeriod.DAY.value:
            key = f"{self.timezone}/day/{start_date.isoformat()}"
        elif self.period == CalendarPeriod.WEEK.value:
            key = (
                f"{self.timezone}/week/{self.week_start}/"
                f"{start_date.isoformat()}"
            )
        else:
            key = f"{self.timezone}/month/{start_date.strftime('%Y-%m')}"

        return CalendarWindow(
            self._midnight(start_date), self._midnight(end_date), key
        )

    def reset_at(self, at: float | datetime | None = None) -> float:
        """Return the UTC epoch second at which the current window resets."""
        return self.get_window(at).reset_at

    def get_expiry(self, at: float | datetime | None = None) -> float:
        """
        :return: seconds from ``at`` until the next calendar boundary.
        """
        at = _timestamp(at)
        if at is None:
            at = time.time()
        return self.get_window(at).reset_at - at

    def key_for(
        self,
        *identifiers: bytes | str | int | float,
        at: float | datetime | None = None,
    ) -> str:
        window = self.get_window(at)
        remainder = "/".join(
            [safe_string(identifier) for identifier in identifiers]
            + [
                safe_string(self.amount),
                safe_string(self.multiples),
                "calendar",
                window.key,
            ]
        )
        return f"{self.namespace}/{remainder}"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, CalendarRateLimitItem)
            and self.namespace == other.namespace
            and self.amount == other.amount
            and self.multiples == other.multiples
            and self.period == other.period
            and self.timezone == other.timezone
            and self.week_start == other.week_start
        )

    def __repr__(self) -> str:
        if self.period == CalendarPeriod.WEEK.value:
            return (
                f"{self.amount} per calendar week starting "
                f"{self.week_start} in {self.timezone}"
            )
        return f"{self.amount} per calendar {self.period} in {self.timezone}"

    def __hash__(self) -> int:
        return hash(
            (
                self.namespace,
                self.amount,
                self.multiples,
                self.period,
                self.timezone,
                self.week_start,
            )
        )


class CalendarRateLimitItemPerDay(CalendarRateLimitItem):
    """A rate limit that resets at local midnight."""

    def __init__(
        self,
        amount: int,
        timezone: str | ZoneInfo,
        multiples: int | None = 1,
        namespace: str = "LIMITER",
    ) -> None:
        super().__init__(
            amount, CalendarPeriod.DAY, timezone, multiples, None, namespace
        )


class CalendarRateLimitItemPerWeek(CalendarRateLimitItem):
    """A rate limit that resets at local midnight on the selected weekday."""

    def __init__(
        self,
        amount: int,
        timezone: str | ZoneInfo,
        multiples: int | None = 1,
        week_start: int | str | None = None,
        namespace: str = "LIMITER",
    ) -> None:
        super().__init__(
            amount, CalendarPeriod.WEEK, timezone, multiples, week_start, namespace
        )


class CalendarRateLimitItemPerMonth(CalendarRateLimitItem):
    """A rate limit that resets at local midnight on the first day."""

    def __init__(
        self,
        amount: int,
        timezone: str | ZoneInfo,
        multiples: int | None = 1,
        namespace: str = "LIMITER",
    ) -> None:
        super().__init__(
            amount, CalendarPeriod.MONTH, timezone, multiples, None, namespace
        )


CalendarRateLimitItem.GRANULARITY = Granularity(0, "calendar")
