"""
Rate limiting with commonly used storage backends
"""

from __future__ import annotations

from . import _version, aio, storage, strategies
from .calendar import (
    CalendarPeriod,
    CalendarRateLimitItem,
    CalendarWindow,
    InvalidTimezoneError,
    RateLimitItemPerCalendarDay,
    RateLimitItemPerCalendarMonth,
    RateLimitItemPerCalendarWeek,
    get_calendar_window,
)
from .limits import (
    RateLimitItem,
    RateLimitItemPerDay,
    RateLimitItemPerHour,
    RateLimitItemPerMinute,
    RateLimitItemPerMonth,
    RateLimitItemPerSecond,
    RateLimitItemPerYear,
)
from .util import WindowStats, parse, parse_many

__all__ = [
    "CalendarPeriod",
    "CalendarRateLimitItem",
    "CalendarWindow",
    "InvalidTimezoneError",
    "RateLimitItem",
    "RateLimitItemPerCalendarDay",
    "RateLimitItemPerCalendarMonth",
    "RateLimitItemPerCalendarWeek",
    "RateLimitItemPerDay",
    "RateLimitItemPerHour",
    "RateLimitItemPerMinute",
    "RateLimitItemPerMonth",
    "RateLimitItemPerSecond",
    "RateLimitItemPerYear",
    "WindowStats",
    "aio",
    "get_calendar_window",
    "parse",
    "parse_many",
    "storage",
    "strategies",
]

__version__ = _version.__version__
