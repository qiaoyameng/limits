from __future__ import annotations

import asyncio
import bisect
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from math import floor, inf

from deprecated.sphinx import versionadded

from limits.combined import (
    CombinedRateLimitRequest,
    CombinedRateLimitResult,
    fixed_window_retry_at,
    prepare_combined_requests,
    sliding_window_retry_at,
)
import limits.typing
from limits.typing import Callable
from limits.aio.storage.base import (
    CombinedWindowSupport,
    MovingWindowSupport,
    SlidingWindowCounterSupport,
    Storage,
)
from limits.storage.base import TimestampedSlidingWindow


class Entry:
    def __init__(self, expiry: int) -> None:
        self.atime = time.time()
        self.expiry = self.atime + expiry


@versionadded(version="2.1")
class MemoryStorage(
    Storage,
    MovingWindowSupport,
    SlidingWindowCounterSupport,
    CombinedWindowSupport,
    TimestampedSlidingWindow,
):
    """
    rate limit storage using :class:`collections.Counter`
    as an in memory storage for fixed & sliding window strategies,
    and a simple list to implement moving window strategy.
    """

    STORAGE_SCHEME = ["async+memory"]
    """
    The storage scheme for in process memory storage for use in an
    async context
    """

    def __init__(
        self,
        uri: str | None = None,
        wrap_exceptions: bool = False,
        clock: Callable[[], float] | None = None,
        **_: str,
    ) -> None:
        self.storage: limits.typing.Counter[str] = Counter()
        self.locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self.expirations: dict[str, float] = {}
        self.events: dict[str, list[Entry]] = {}
        self.clock = clock or time.time
        self._custom_clock = clock is not None
        self._state_lock: asyncio.Lock | None = None
        self.timer: asyncio.Task[None] | None = None
        super().__init__(uri, wrap_exceptions=wrap_exceptions, **_)

    async def _get_state_lock(self) -> asyncio.Lock:
        if self._state_lock is None:
            self._state_lock = asyncio.Lock()
        return self._state_lock

    def __getstate__(self) -> dict[str, limits.typing.Any]:  # type: ignore[explicit-any]
        state = self.__dict__.copy()
        del state["timer"]
        del state["locks"]
        del state["_state_lock"]
        return state

    def __setstate__(self, state: dict[str, limits.typing.Any]) -> None:  # type: ignore[explicit-any]
        self.__dict__.update(state)
        self.timer = None
        self._state_lock = None
        self.locks = defaultdict(asyncio.Lock)
        asyncio.ensure_future(self.__schedule_expiry())

    async def __expire_events(self) -> None:
        try:
            now = time.time()
            for key in list(self.events.keys()):
                async with self.locks[key]:
                    cutoff = await asyncio.to_thread(
                        lambda evts: bisect.bisect_left(
                            evts, -now, key=lambda event: -event.expiry
                        ),
                        self.events[key],
                    )
                    if self.events.get(key, []):
                        self.events[key] = self.events[key][:cutoff]
                    if not self.events.get(key, None):
                        self.events.pop(key, None)
                        self.locks.pop(key, None)

            if not self._custom_clock:
                now = time.time()
                async with await self._get_state_lock():
                    for key in list(self.expirations.keys()):
                        if self.expirations.get(key, inf) <= now:
                            self.storage.pop(key, None)
                            self.expirations.pop(key, None)
                            self.locks.pop(key, None)
        except asyncio.CancelledError:
            return

    async def __schedule_expiry(self) -> None:
        if not self.timer or self.timer.done():
            self.timer = asyncio.create_task(self.__expire_events())

    @property
    def base_exceptions(
        self,
    ) -> type[Exception] | tuple[type[Exception], ...]:  # pragma: no cover
        return ValueError

    async def incr(self, key: str, expiry: float, amount: int = 1) -> int:
        """
        increments the counter for a given rate limit key

        :param key: the key to increment
        :param expiry: amount in seconds for the key to expire in
        :param amount: the number to increment by
        """
        await self.get(key)
        await self.__schedule_expiry()
        async with await self._get_state_lock(), self.locks[key]:
            await self._get_locked(key, self.clock())
            self.storage[key] += amount
            if self.storage[key] == amount:
                self.expirations[key] = self.clock() + expiry
        return self.storage.get(key, amount)

    async def decr(self, key: str, amount: int = 1) -> int:
        """
        decrements the counter for a given rate limit key. 0 is the minimum allowed value.

        :param amount: the number to increment by
        """
        await self.get(key)
        await self.__schedule_expiry()
        async with await self._get_state_lock(), self.locks[key]:
            self.storage[key] = max(self.storage[key] - amount, 0)

        return self.storage.get(key, amount)

    async def get(self, key: str) -> int:
        """
        :param key: the key to get the counter value for
        """
        async with await self._get_state_lock():
            await self._get_locked(key, self.clock())

        return self.storage.get(key, 0)

    async def clear(self, key: str) -> None:
        """
        :param key: the key to clear rate limits for
        """
        async with await self._get_state_lock(), self.locks[key]:
            self.storage.pop(key, None)
            self.expirations.pop(key, None)
            self.events.pop(key, None)
            self.locks.pop(key, None)

    async def acquire_entry(
        self, key: str, limit: int, expiry: int, amount: int = 1
    ) -> bool:
        """
        :param key: rate limit key to acquire an entry in
        :param limit: amount of entries allowed
        :param expiry: expiry of the entry
        :param amount: the number of entries to acquire
        """
        if amount > limit:
            return False

        await self.__schedule_expiry()
        async with self.locks[key]:
            self.events.setdefault(key, [])
            timestamp = time.time()
            try:
                entry: Entry | None = self.events[key][limit - amount]
            except IndexError:
                entry = None

            if entry and entry.atime >= timestamp - expiry:
                return False
            else:
                self.events[key][:0] = [Entry(expiry)] * amount
            return True

    async def get_expiry(self, key: str) -> float:
        """
        :param key: the key to get the expiry for
        """

        async with await self._get_state_lock():
            return self.expirations.get(key, self.clock())

    async def get_moving_window(
        self, key: str, limit: int, expiry: int
    ) -> tuple[float, int]:
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

    async def acquire_sliding_window_entry(
        self,
        key: str,
        limit: int,
        expiry: int,
        amount: int = 1,
    ) -> bool:
        if amount > limit:
            return False
        async with await self._get_state_lock():
            return await self._acquire_sliding_window_entry_locked(
                key, limit, expiry, amount, self.clock()
            )

    async def _acquire_sliding_window_entry_locked(
        self,
        key: str,
        limit: int,
        expiry: int,
        amount: int,
        now: float,
    ) -> bool:
        previous_key, current_key = self.sliding_window_keys(key, expiry, now)
        (
            previous_count,
            previous_ttl,
            current_count,
            _,
        ) = await self._get_sliding_window_info(previous_key, current_key, expiry, now)
        weighted_count = previous_count * previous_ttl / expiry + current_count
        if floor(weighted_count) + amount > limit:
            return False
        else:
            # Hit, increase the current counter.
            # If the counter doesn't exist yet, set twice the theorical expiry.
            self.storage[current_key] += amount
            if self.storage[current_key] == amount:
                self.expirations[current_key] = now + 2 * expiry
            current_count = self.storage[current_key]
            weighted_count = previous_count * previous_ttl / expiry + current_count
            if floor(weighted_count) > limit:
                # Another hit won the race condition: revert the incrementation and refuse this hit
                # Limitation: during high concurrency at the end of the window,
                # the counter is shifted and cannot be decremented, so less requests than expected are allowed.
                self.storage[current_key] = max(
                    self.storage[current_key] - amount, 0
                )
                return False
            return True

    async def get_sliding_window(
        self, key: str, expiry: int
    ) -> tuple[int, float, int, float]:
        async with await self._get_state_lock():
            now = self.clock()
            previous_key, current_key = self.sliding_window_keys(key, expiry, now)
            return await self._get_sliding_window_info(
                previous_key, current_key, expiry, now
            )

    async def clear_sliding_window(self, key: str, expiry: int) -> None:
        now = self.clock()
        previous_key, current_key = self.sliding_window_keys(key, expiry, now)
        await self.clear(current_key)
        await self.clear(previous_key)

    async def _get_sliding_window_info(
        self,
        previous_key: str,
        current_key: str,
        expiry: int,
        now: float,
    ) -> tuple[int, float, int, float]:
        previous_count = await self._get_locked(previous_key, now)
        current_count = await self._get_locked(current_key, now)
        if previous_count == 0:
            previous_ttl = float(0)
        else:
            previous_ttl = (1 - (((now - expiry) / expiry) % 1)) * expiry
        current_ttl = (1 - ((now / expiry) % 1)) * expiry + expiry
        return previous_count, previous_ttl, current_count, current_ttl

    async def _get_locked(self, key: str, now: float) -> int:
        if self.expirations.get(key, 0) <= now:
            self.storage.pop(key, None)
            self.expirations.pop(key, None)
            self.locks.pop(key, None)
        return self.storage.get(key, 0)

    async def check(self) -> bool:
        """
        check if storage is healthy
        """

        return True

    async def reset(self) -> int | None:
        async with await self._get_state_lock():
            num_items = max(len(self.storage), len(self.events))
            self.storage.clear()
            self.expirations.clear()
            self.events.clear()
            self.locks.clear()

        return num_items

    async def acquire_many(
        self, requests: Sequence[CombinedRateLimitRequest]
    ) -> CombinedRateLimitResult:
        now = self.clock()
        groups = prepare_combined_requests(requests, now)
        rejected = []
        retry_at = inf
        await self.__schedule_expiry()

        async with await self._get_state_lock():
            snapshots = []
            for group in groups:
                if group.window == "fixed-window":
                    current_count = await self._get_locked(group.key, now)
                    snapshots.append((group, current_count, 0, 0.0))
                    if current_count + group.cost > group.limit:
                        rejected.extend(
                            requests[index] for index in group.sources
                        )
                        retry_at = min(
                            retry_at,
                            fixed_window_retry_at(
                                current_count,
                                group.cost,
                                group.limit,
                                self.expirations.get(group.key, now),
                            ),
                        )
                else:
                    previous_key, current_key = self.sliding_window_keys(
                        group.key, group.expiry, now
                    )
                    previous_count, previous_ttl, current_count, _ = (
                        await self._get_sliding_window_info(
                            previous_key, current_key, group.expiry, now
                        )
                    )
                    snapshots.append(
                        (group, 0, previous_count, previous_ttl, current_count)
                    )
                    weighted_count = (
                        previous_count * previous_ttl / group.expiry + current_count
                    )
                    if floor(weighted_count) + group.cost > group.limit:
                        rejected.extend(
                            requests[index] for index in group.sources
                        )
                        retry_at = min(
                            retry_at,
                            sliding_window_retry_at(
                                now,
                                previous_count,
                                previous_ttl,
                                current_count,
                                group.cost,
                                group.limit,
                                group.expiry,
                            ),
                        )

            if rejected:
                return CombinedRateLimitResult(False, tuple(rejected), retry_at)

            for snapshot in snapshots:
                group = snapshot[0]
                if group.window == "fixed-window":
                    current_count = snapshot[1]
                    self.storage[group.key] = current_count + group.cost
                    if current_count == 0:
                        self.expirations[group.key] = now + group.expiry
                else:
                    _, _, _, _, current_count = snapshot
                    _, current_key = self.sliding_window_keys(
                        group.key, group.expiry, now
                    )
                    self.storage[current_key] = current_count + group.cost
                    if current_count == 0:
                        self.expirations[current_key] = now + 2 * group.expiry

        return CombinedRateLimitResult(True, (), now)

    def __del__(self) -> None:
        try:
            if self.timer and not self.timer.done():
                self.timer.cancel()
        except RuntimeError:  # noqa
            pass
