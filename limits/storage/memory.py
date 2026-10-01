from __future__ import annotations

import bisect
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from math import floor, inf, isfinite
from typing import Literal, Sequence

import limits.typing
from limits.storage.base import (
    MovingWindowSupport,
    SlidingWindowCounterSupport,
    Storage,
    TimestampedSlidingWindow,
    _wrap_errors,
)


class Entry:
    def __init__(self, expiry: float) -> None:
        self.atime = time.time()
        self.expiry = self.atime + expiry


@dataclass(frozen=True)
class CombinedWindowRequest:
    key: str
    limit: int
    expiry: int
    cost: int = 1
    strategy: Literal["fixed-window", "sliding-window-counter"] = "fixed-window"
    source: int | str | None = None


@dataclass(frozen=True)
class CombinedWindowRejection:
    source: int | str | None
    key: str
    retry_at: float | None


@dataclass(frozen=True)
class CombinedWindowResult:
    allowed: bool
    rejected: tuple[CombinedWindowRejection, ...] = ()
    retry_at: float | None = None


@dataclass(frozen=True)
class _MergedWindowRequest:
    key: str
    logical_key: str
    limit: int
    expiry: int
    cost: int
    strategy: str
    sources: tuple[int | str, ...]


class MemoryStorage(
    Storage, MovingWindowSupport, SlidingWindowCounterSupport, TimestampedSlidingWindow
):
    """
    rate limit storage using :class:`collections.Counter`
    as an in memory storage for fixed and sliding window strategies,
    and a simple list to implement moving window strategy.

    """

    STORAGE_SCHEME = ["memory"]

    def __init__(self, uri: str | None = None, wrap_exceptions: bool = False, **_: str):
        self.storage: limits.typing.Counter[str] = Counter()
        self.locks: defaultdict[str, threading.RLock] = defaultdict(threading.RLock)
        self._transaction_lock = threading.RLock()
        self.expirations: dict[str, float] = {}
        self.events: dict[str, list[Entry]] = {}
        self._timer_lock = threading.Lock()
        self.timer: threading.Timer = threading.Timer(0.01, self.__expire_events)
        self.timer.start()
        super().__init__(uri, wrap_exceptions=wrap_exceptions, **_)

    def __getstate__(self) -> dict[str, limits.typing.Any]:  # type: ignore[explicit-any]
        state = self.__dict__.copy()
        del state["timer"]
        del state["locks"]
        del state["_transaction_lock"]
        del state["_timer_lock"]
        return state

    def __setstate__(self, state: dict[str, limits.typing.Any]) -> None:  # type: ignore[explicit-any]
        self.__dict__.update(state)
        self.locks = defaultdict(threading.RLock)
        self._transaction_lock = threading.RLock()
        self._timer_lock = threading.Lock()
        self.timer = threading.Timer(0.01, self.__expire_events)
        self.timer.start()

    def __expire_events(self) -> None:
        with self._transaction_lock:
            for key in list(self.events.keys()):
                with self.locks[key]:
                    if events := self.events.get(key, []):
                        oldest = bisect.bisect_left(
                            events, -time.time(), key=lambda event: -event.expiry
                        )
                        self.events[key] = self.events[key][:oldest]
                    if not self.events.get(key, None):
                        self.locks.pop(key, None)
            for key in list(self.expirations.keys()):
                if self.expirations[key] <= time.time():
                    self.storage.pop(key, None)
                    self.expirations.pop(key, None)
                    self.locks.pop(key, None)

    def __schedule_expiry(self) -> None:
        with self._timer_lock:
            if not self.timer.is_alive():
                self.timer = threading.Timer(0.01, self.__expire_events)
                self.timer.start()

    @property
    def base_exceptions(
        self,
    ) -> type[Exception] | tuple[type[Exception], ...]:  # pragma: no cover
        return ValueError

    def incr(self, key: str, expiry: float, amount: int = 1) -> int:
        """
        increments the counter for a given rate limit key

        :param key: the key to increment
        :param expiry: amount in seconds for the key to expire in
        :param amount: the number to increment by
        """
        with self._transaction_lock:
            self.get(key)
            self.__schedule_expiry()
            with self.locks[key]:
                self.storage[key] += amount
                if self.storage[key] == amount:
                    self.expirations[key] = time.time() + expiry
            return self.storage.get(key, 0)

    def decr(self, key: str, amount: int = 1) -> int:
        """
        decrements the counter for a given rate limit key

        :param key: the key to decrement
        :param amount: the number to decrement by
        """
        with self._transaction_lock:
            self.get(key)
            self.__schedule_expiry()
            with self.locks[key]:
                self.storage[key] = max(self.storage[key] - amount, 0)

            return self.storage.get(key, 0)

    def get(self, key: str) -> int:
        """
        :param key: the key to get the counter value for
        """

        with self._transaction_lock:
            if self.expirations.get(key, 0) <= time.time():
                self.storage.pop(key, None)
                self.expirations.pop(key, None)
                self.locks.pop(key, None)

            return self.storage.get(key, 0)

    def clear(self, key: str) -> None:
        """
        :param key: the key to clear rate limits for
        """
        with self._transaction_lock:
            self.storage.pop(key, None)
            self.expirations.pop(key, None)
            self.events.pop(key, None)
            self.locks.pop(key, None)

    def acquire_entry(self, key: str, limit: int, expiry: int, amount: int = 1) -> bool:
        """
        :param key: rate limit key to acquire an entry in
        :param limit: amount of entries allowed
        :param expiry: expiry of the entry
        :param amount: the number of entries to acquire
        """
        if amount > limit:
            return False

        with self._transaction_lock:
            self.__schedule_expiry()
            with self.locks[key]:
                self.events.setdefault(key, [])
                timestamp = time.time()
                try:
                    entry = self.events[key][limit - amount]
                except IndexError:
                    entry = None

                if entry and entry.atime >= timestamp - expiry:
                    return False
                else:
                    self.events[key][:0] = [Entry(expiry)] * amount
                    return True

    def get_expiry(self, key: str) -> float:
        """
        :param key: the key to get the expiry for
        """

        return self.expirations.get(key, time.time())

    def get_moving_window(self, key: str, limit: int, expiry: int) -> tuple[float, int]:
        """
        returns the starting point and the number of entries in the moving
        window

        :param key: rate limit key
        :param expiry: expiry of entry
        :return: (start of window, number of acquired entries)
        """
        timestamp = time.time()
        if events := self.events.get(key, []):
            oldest = bisect.bisect_left(
                events, -(timestamp - expiry), key=lambda entry: -entry.atime
            )
            return events[oldest - 1].atime, oldest
        return timestamp, 0

    def acquire_sliding_window_entry(
        self,
        key: str,
        limit: int,
        expiry: int,
        amount: int = 1,
    ) -> bool:
        if amount > limit:
            return False
        with self._transaction_lock:
            now = time.time()
            previous_key, current_key = self.sliding_window_keys(key, expiry, now)
            previous_count = self.get(previous_key)
            current_count = self.get(current_key)
            if previous_count == 0:
                previous_ttl = 0.0
            else:
                previous_ttl = (1 - (((now - expiry) / expiry) % 1)) * expiry
            weighted_count = previous_count * previous_ttl / expiry + current_count
            if floor(weighted_count) + amount > limit:
                return False

            self.storage[current_key] += amount
            if self.storage[current_key] == amount:
                self.expirations[current_key] = now + 2 * expiry
            weighted_count = (
                previous_count * previous_ttl / expiry + self.storage[current_key]
            )
            if floor(weighted_count) > limit:
                self.storage[current_key] = max(
                    self.storage[current_key] - amount, 0
                )
                return False
            return True

    def _get_sliding_window_info(
        self,
        previous_key: str,
        current_key: str,
        expiry: int,
        now: float,
    ) -> tuple[int, float, int, float]:
        previous_count = self.get(previous_key)
        current_count = self.get(current_key)
        if previous_count == 0:
            previous_ttl = float(0)
        else:
            previous_ttl = (1 - (((now - expiry) / expiry) % 1)) * expiry
        current_ttl = (1 - ((now / expiry) % 1)) * expiry + expiry
        return previous_count, previous_ttl, current_count, current_ttl

    def get_sliding_window(
        self, key: str, expiry: int
    ) -> tuple[int, float, int, float]:
        now = time.time()
        previous_key, current_key = self.sliding_window_keys(key, expiry, now)
        return self._get_sliding_window_info(previous_key, current_key, expiry, now)

    def clear_sliding_window(self, key: str, expiry: int) -> None:
        now = time.time()
        previous_key, current_key = self.sliding_window_keys(key, expiry, now)
        self.clear(previous_key)
        self.clear(current_key)

    @_wrap_errors
    def acquire_combined_entries(
        self, requests: Sequence[CombinedWindowRequest]
    ) -> CombinedWindowResult:
        merged_requests = self._merge_combined_requests(requests)
        with self._transaction_lock:
            now = time.time()
            physical_requests = self._resolve_physical_requests(merged_requests, now)
            physical_keys = sorted({item.key for item in physical_requests})
            for physical_key in physical_keys:
                self.locks[physical_key].acquire()
            try:
                self._check_physical_requests(physical_requests)
                rejections: list[CombinedWindowRejection] = []
                retry_at: float | None = None

                for request in merged_requests:
                    request_retry_at = self._get_combined_retry_at(
                        request, now, self.storage, self.expirations
                    )
                    if request_retry_at is not None:
                        if request_retry_at < (retry_at or inf):
                            retry_at = request_retry_at
                        for source in request.sources:
                            rejections.append(
                                CombinedWindowRejection(
                                    source,
                                    request.key,
                                    request_retry_at,
                                )
                            )

                if rejections:
                    return self._combined_result(rejections, retry_at)

                self.__schedule_expiry()
                for request in physical_requests:
                    self._commit_physical_request(
                        request, now, self.storage, self.expirations
                    )
                return CombinedWindowResult(True)
            finally:
                for physical_key in reversed(physical_keys):
                    self.locks[physical_key].release()

    @staticmethod
    def _combined_result(
        rejections: Sequence[CombinedWindowRejection], retry_at: float | None
    ) -> CombinedWindowResult:
        public_retry_at = (
            retry_at if retry_at is not None and isfinite(retry_at) else None
        )
        return CombinedWindowResult(
            False,
            tuple(
                CombinedWindowRejection(
                    rejection.source,
                    rejection.key,
                    rejection.retry_at
                    if rejection.retry_at is not None
                    and isfinite(rejection.retry_at)
                    else None,
                )
                for rejection in rejections
            ),
            public_retry_at,
        )

    @staticmethod
    def _merge_combined_requests(
        requests: Sequence[CombinedWindowRequest],
    ) -> tuple[_MergedWindowRequest, ...]:
        if not requests:
            raise ValueError("at least one rate limit request is required")

        merged: dict[tuple[str, str, int, int], _MergedWindowRequest] = {}
        sources_by_request: dict[tuple[str, str, int, int], list[int | str]] = {}

        for index, request in enumerate(requests):
            if not isinstance(request, CombinedWindowRequest):
                raise ValueError(
                    "combined rate limit requests must be CombinedWindowRequest"
                )
            if not isinstance(request.key, str) or not request.key:
                raise ValueError("rate limit key must be a non-empty string")
            if request.strategy not in {"fixed-window", "sliding-window-counter"}:
                raise ValueError(f"unsupported rate limit strategy: {request.strategy}")
            if type(request.limit) is not int or request.limit <= 0:
                raise ValueError("rate limit must be a positive integer")
            if type(request.expiry) is not int or request.expiry <= 0:
                raise ValueError("rate limit expiry must be a positive integer")
            if type(request.cost) is not int or request.cost <= 0:
                raise ValueError("rate limit cost must be a positive integer")

            source = index if request.source is None else request.source
            identity = (request.strategy, request.key, request.limit, request.expiry)
            sources = sources_by_request.setdefault(identity, [])
            sources.append(source)
            existing = merged.get(identity)
            if existing is None:
                merged[identity] = _MergedWindowRequest(
                    request.key,
                    request.key,
                    request.limit,
                    request.expiry,
                    request.cost,
                    request.strategy,
                    (source,),
                )
            else:
                merged[identity] = _MergedWindowRequest(
                    existing.key,
                    existing.logical_key,
                    existing.limit,
                    existing.expiry,
                    existing.cost + request.cost,
                    existing.strategy,
                    tuple(sources),
                )

        return tuple(merged.values())

    def _resolve_physical_requests(
        self,
        requests: Sequence[_MergedWindowRequest],
        now: float,
    ) -> tuple[_MergedWindowRequest, ...]:
        physical_requests: list[_MergedWindowRequest] = []
        for request in requests:
            if request.strategy == "fixed-window":
                physical_requests.append(request)
            else:
                previous_key, current_key = self.sliding_window_keys(
                    request.key, request.expiry, now
                )
                physical_requests.extend(
                    (
                        _MergedWindowRequest(
                            previous_key,
                            request.logical_key,
                            request.limit,
                            request.expiry,
                            0,
                            "sliding-previous",
                            request.sources,
                        ),
                        _MergedWindowRequest(
                            current_key,
                            request.logical_key,
                            request.limit,
                            request.expiry,
                            request.cost,
                            "sliding-current",
                            request.sources,
                        ),
                    )
                )
        return tuple(physical_requests)

    @staticmethod
    def _check_physical_requests(requests: Sequence[_MergedWindowRequest]) -> None:
        owners: dict[str, tuple[str, int, int]] = {}
        for request in requests:
            family = (
                "sliding-window-counter"
                if request.strategy.startswith("sliding")
                else "fixed-window"
            )
            identity = (
                family,
                request.logical_key,
                request.limit,
                request.expiry,
            )
            existing = owners.get(request.key)
            if existing is not None and existing != identity:
                raise ValueError(
                    f"combined rate limit requests share storage key {request.key}"
                )
            owners[request.key] = identity

    @staticmethod
    def _get_combined_retry_at(
        request: _MergedWindowRequest,
        now: float,
        storage: Counter[str],
        expirations: dict[str, float],
    ) -> float | None:
        if request.strategy == "fixed-window":
            count = MemoryStorage._get_active_fixed_count(
                storage, expirations, request.key, now
            )
            if count + request.cost <= request.limit:
                return None
            if request.cost > request.limit:
                return inf
            return expirations.get(request.key, now + request.expiry)

        previous_key, current_key = TimestampedSlidingWindow.sliding_window_keys(
            request.key, request.expiry, now
        )
        previous_count = MemoryStorage._get_active_fixed_count(
            storage, expirations, previous_key, now
        )
        current_count = MemoryStorage._get_active_fixed_count(
            storage, expirations, current_key, now
        )
        previous_ttl = request.expiry - (now % request.expiry)
        weighted_count = previous_count * previous_ttl / request.expiry + current_count
        if floor(weighted_count) + request.cost <= request.limit:
            return None

        threshold = request.limit - request.cost + 1
        if threshold <= 0:
            return inf
        if current_count >= threshold:
            current_window = floor(now / request.expiry)
            return (current_window + 2) * request.expiry

        boundary = (floor(now / request.expiry) + 1) * request.expiry
        if previous_count == 0:
            return boundary
        target = threshold - current_count
        decay_deadline = previous_ttl - request.expiry * target / previous_count
        if 0 < decay_deadline < previous_ttl:
            return now + decay_deadline
        return boundary

    @staticmethod
    def _get_active_fixed_count(
        storage: Counter[str],
        expirations: dict[str, float],
        key: str,
        now: float,
    ) -> int:
        expiry = expirations.get(key)
        if expiry is not None and expiry <= now:
            return 0
        return storage.get(key, 0)

    @staticmethod
    def _commit_physical_request(
        request: _MergedWindowRequest,
        now: float,
        storage: Counter[str],
        expirations: dict[str, float],
    ) -> None:
        if request.strategy not in {"fixed-window", "sliding-current"}:
            return

        expiry = expirations.get(request.key)
        if expiry is None or expiry <= now:
            storage[request.key] = request.cost
            expirations[request.key] = (
                now + request.expiry
                if request.strategy == "fixed-window"
                else now + 2 * request.expiry
            )
        else:
            storage[request.key] += request.cost

    def check(self) -> bool:
        """
        check if storage is healthy
        """

        return True

    def reset(self) -> int | None:
        with self._transaction_lock:
            num_items = max(len(self.storage), len(self.events))
            self.storage.clear()
            self.expirations.clear()
            self.events.clear()
            self.locks.clear()
            return num_items
