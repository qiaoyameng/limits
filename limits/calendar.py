from __future__ import annotations

import time
from datetime import date, datetime, timedelta
from enum import Enum
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from limits.limits import RateLimitItem, safe_string
from limits.typing import NamedTuple


class CalendarPeriod(str, Enum):
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


class CalendarWindow(NamedTuple):
    start: float
    end: float


class InvalidTimezoneError(ValueError, ZoneInfoNotFoundError):
    """Raised when an IANA timezone name cannot be resolved."""


CalendarTime = float | int | datetime


def reject_non_fixed_window(item: RateLimitItem) -> None:
    if isinstance(item, CalendarRateLimitItem):
        raise NotImplementedError(
            "Calendar rate limits are only supported by the fixed-window strategy"
        )


def _coerce_timezone(
    timezone: str | ZoneInfo,
) -> tuple[ZoneInfo, str]:
    if isinstance(timezone, ZoneInfo):
        return timezone, str(timezone)
    if isinstance(timezone, str):
        try:
            zone = ZoneInfo(timezone)
        except ZoneInfoNotFoundError as error:
            raise InvalidTimezoneError(
                f"Unknown IANA timezone: {timezone}"
            ) from error
        return zone, timezone
    raise ValueError("timezone must be an IANA timezone name or ZoneInfo")


def _coerce_period(period: CalendarPeriod | str) -> CalendarPeriod:
    try:
        return CalendarPeriod(period)
    except ValueError as error:
        raise ValueError("period must be one of: day, week, month") from error


def _coerce_at(at: CalendarTime | None) -> float | None:
    if at is None:
        return None
    if isinstance(at, datetime):
        if at.tzinfo is None:
            raise ValueError("datetime must be timezone-aware")
        return at.timestamp()
    return float(at)


def _first_moment_of_date(day: date, timezone: ZoneInfo) -> float:
    naive_midnight = datetime(day.year, day.month, day.day, tzinfo=timezone)
    candidates = [
        naive_midnight.replace(fold=0),
        naive_midnight.replace(fold=1),
    ]
    valid_candidates = [
        candidate
        for candidate in candidates
        if datetime.fromtimestamp(candidate.timestamp(), timezone).date() == day
    ]
    return min(candidate.timestamp() for candidate in valid_candidates)


def _period_start_date(
    day: date, period: CalendarPeriod, week_start: int
) -> date:
    if period is CalendarPeriod.DAY:
        return day
    if period is CalendarPeriod.WEEK:
        return day - timedelta(days=(day.weekday() - week_start) % 7)
    return day.replace(day=1)


def _next_period_start_date(
    start: date, period: CalendarPeriod
) -> date:
    if period is CalendarPeriod.MONTH:
        return date(
            start.year + start.month // 12,
            start.month % 12 + 1,
            1,
        )
    days = 7 if period is CalendarPeriod.WEEK else 1
    return start + timedelta(days=days)


def get_calendar_window(
    period: CalendarPeriod,
    timezone: ZoneInfo,
    week_start: int = 0,
    at: CalendarTime | None = None,
) -> CalendarWindow:
    at = _coerce_at(at)
    if at is None:
        at = time.time()

    local_day = datetime.fromtimestamp(at, timezone).date()
    start_day = _period_start_date(local_day, period, week_start)
    end_day = _next_period_start_date(start_day, period)

    return CalendarWindow(
        _first_moment_of_date(start_day, timezone),
        _first_moment_of_date(end_day, timezone),
    )


class CalendarRateLimitItem(RateLimitItem):
    """
    A fixed-window rate limit aligned to a calendar boundary in an IANA timezone.
    """

    __slots__ = [
        "period",
        "timezone",
        "timezone_name",
        "week_start",
    ]

    def __init__(
        self,
        amount: int,
        period: CalendarPeriod | str,
        timezone: str | ZoneInfo,
        *,
        week_start: int = 0,
        multiples: int = 1,
        namespace: str = "LIMITER",
    ) -> None:
        self.period = _coerce_period(period)
        self.timezone, self.timezone_name = _coerce_timezone(timezone)
        if not isinstance(week_start, int) or isinstance(week_start, bool):
            raise ValueError("week_start must be an integer between 0 and 6")
        if not 0 <= week_start <= 6:
            raise ValueError("week_start must be an integer between 0 and 6")
        if (
            not isinstance(multiples, int)
            or isinstance(multiples, bool)
            or multiples != 1
        ):
            raise ValueError("calendar rate limits cannot use multiples")

        self.week_start = week_start
        super().__init__(amount, 1, namespace)

    def get_window(self, at: CalendarTime | None = None) -> CalendarWindow:
        return get_calendar_window(self.period, self.timezone, self.week_start, at)

    def get_reset_at(self, at: CalendarTime | None = None) -> float:
        return self.get_window(at).end

    def get_expiry(self, at: CalendarTime | None = None) -> float:
        at = _coerce_at(at)
        if at is None:
            at = time.time()
        return self.get_window(at).end - at

    def _period_identity(self, window: CalendarWindow) -> str:
        start_day = datetime.fromtimestamp(window.start, self.timezone).date()
        if self.period is CalendarPeriod.WEEK:
            iso_year, iso_week, _ = start_day.isocalendar()
            return f"{iso_year}-W{iso_week:02d}-w{self.week_start}"
        if self.period is CalendarPeriod.MONTH:
            return start_day.strftime("%Y-%m")
        return start_day.isoformat()

    def key_for(
        self, *identifiers: bytes | str | int | float, at: CalendarTime | None = None
    ) -> str:
        window = self.get_window(at)
        remainder = "/".join(
            [safe_string(identifier) for identifier in identifiers]
            + [
                safe_string(self.amount),
                "calendar",
                self.period.value,
                self.timezone_name,
                self._period_identity(window),
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
            and str(self.timezone) == str(other.timezone)
            and self.week_start == other.week_start
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.namespace,
                self.amount,
                self.period,
                str(self.timezone),
                self.week_start,
            )
        )

    def __lt__(self, other: RateLimitItem) -> bool:
        if not isinstance(other, CalendarRateLimitItem):
            return NotImplemented
        return self.get_reset_at() < other.get_reset_at()

    def __repr__(self) -> str:
        if self.period is CalendarPeriod.WEEK:
            return (
                f"{self.amount} per calendar week starting "
                f"{self.week_start} in {self.timezone_name}"
            )
        return f"{self.amount} per calendar {self.period.value} in {self.timezone_name}"


class RateLimitItemPerCalendarDay(CalendarRateLimitItem):
    def __init__(
        self,
        amount: int,
        timezone: str | ZoneInfo,
        *,
        multiples: int = 1,
        namespace: str = "LIMITER",
    ) -> None:
        super().__init__(
            amount,
            CalendarPeriod.DAY,
            timezone,
            multiples=multiples,
            namespace=namespace,
        )


class RateLimitItemPerCalendarWeek(CalendarRateLimitItem):
    def __init__(
        self,
        amount: int,
        timezone: str | ZoneInfo,
        week_start: int = 0,
        *,
        multiples: int = 1,
        namespace: str = "LIMITER",
    ) -> None:
        super().__init__(
            amount,
            CalendarPeriod.WEEK,
            timezone,
            week_start=week_start,
            multiples=multiples,
            namespace=namespace,
        )


class RateLimitItemPerCalendarMonth(CalendarRateLimitItem):
    def __init__(
        self,
        amount: int,
        timezone: str | ZoneInfo,
        *,
        multiples: int = 1,
        namespace: str = "LIMITER",
    ) -> None:
        super().__init__(
            amount,
            CalendarPeriod.MONTH,
            timezone,
            multiples=multiples,
            namespace=namespace,
        )
