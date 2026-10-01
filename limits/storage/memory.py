from __future__ import annotations

import bisect
import threading
import time
import uuid
from collections import Counter, defaultdict
from math import floor

import limits.typing
from limits.errors import LeaseConflictError, LeaseExpiredError
from limits.storage.base import (
    MovingWindowSupport,
    ReservationSupport,
    SlidingWindowCounterSupport,
    Storage,
    TimestampedSlidingWindow,
)
from limits.util import Lease


class Entry:
    def __init__(self, expiry: float) -> None:
        self.atime = time.time()
        self.expiry = self.atime + expiry


class LeaseRecord:
    """
    Internal state tracked by the storage for an outstanding lease.
    """

    def __init__(self, lease: Lease) -> None:
        self.lease_id = lease.lease_id
        self.key = lease.key
        self.amount = lease.amount
        self.expires_at = lease.expires_at
        self.generation = lease.generation
        self.state = "open"
        self.committed_cost: int | None = None


class MemoryStorage(
    Storage,
    MovingWindowSupport,
    SlidingWindowCounterSupport,
    ReservationSupport,
    TimestampedSlidingWindow,
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
        self.expirations: dict[str, float] = {}
        self.events: dict[str, list[Entry]] = {}
        self.leases: dict[str, LeaseRecord] = {}
        self.window_generations: dict[str, int] = {}
        self._timer_lock = threading.Lock()
        self.timer: threading.Timer = threading.Timer(0.01, self.__expire_events)
        self.timer.start()
        super().__init__(uri, wrap_exceptions=wrap_exceptions, **_)

    def __getstate__(self) -> dict[str, limits.typing.Any]:  # type: ignore[explicit-any]
        state = self.__dict__.copy()
        del state["timer"]
        del state["locks"]
        del state["_timer_lock"]
        return state

    def __setstate__(self, state: dict[str, limits.typing.Any]) -> None:  # type: ignore[explicit-any]
        self.__dict__.update(state)
        self.locks = defaultdict(threading.RLock)
        self._timer_lock = threading.Lock()
        self.timer = threading.Timer(0.01, self.__expire_events)
        self.timer.start()

    def __expire_events(self) -> None:
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
        for lease_id in list(self.leases.keys()):
            record = self.leases.get(lease_id)
            if record is None:
                continue
            with self.locks[record.key]:
                self._expire_lease(record, time.time())
                if record.state != "open" and record.expires_at <= time.time():
                    self.leases.pop(lease_id, None)

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
        self.get(key)
        self.__schedule_expiry()
        with self.locks[key]:
            self.storage[key] += amount
            if self.storage[key] == amount:
                self.expirations[key] = time.time() + expiry
                self.window_generations[key] = self.window_generations.get(key, 0) + 1
        return self.storage.get(key, 0)

    def decr(self, key: str, amount: int = 1) -> int:
        """
        decrements the counter for a given rate limit key

        :param key: the key to decrement
        :param amount: the number to decrement by
        """
        self.get(key)
        self.__schedule_expiry()
        with self.locks[key]:
            self.storage[key] = max(self.storage[key] - amount, 0)

        return self.storage.get(key, 0)

    def get(self, key: str) -> int:
        """
        :param key: the key to get the counter value for
        """

        if self.expirations.get(key, 0) <= time.time():
            self.storage.pop(key, None)
            self.expirations.pop(key, None)
            self.locks.pop(key, None)

        return self.storage.get(key, 0)

    def clear(self, key: str) -> None:
        """
        :param key: the key to clear rate limits for
        """
        self.storage.pop(key, None)
        self.expirations.pop(key, None)
        self.events.pop(key, None)
        self.locks.pop(key, None)
        self.window_generations[key] = self.window_generations.get(key, 0) + 1

    def _window_live(self, record: LeaseRecord, now: float) -> bool:
        """
        whether the window a lease was issued in is still the current one
        """
        return (
            self.window_generations.get(record.key, 0) == record.generation
            and self.expirations.get(record.key, 0) > now
        )

    def _refund(self, record: LeaseRecord, amount: int) -> None:
        if amount > 0:
            self.storage[record.key] = max(self.storage.get(record.key, 0) - amount, 0)

    def _expire_lease(self, record: LeaseRecord, now: float) -> None:
        """
        release an open lease whose ttl has elapsed, refunding the reservation
        only if the window it was taken from is still current
        """
        if record.state == "open" and record.expires_at <= now:
            if self._window_live(record, now):
                self._refund(record, record.amount)
            record.state = "expired"

    def reserve(
        self,
        key: str,
        limit: int,
        expiry: int,
        amount: int = 1,
        lease_ttl: float | None = None,
    ) -> Lease | None:
        """
        :param key: rate limit key to reserve against
        :param limit: amount of entries allowed
        :param expiry: expiry of the window
        :param amount: the amount to reserve
        :param lease_ttl: how many seconds the lease remains valid for
        :return: the lease, or ``None`` if the reservation would exceed the limit
        """
        if amount > limit:
            return None

        self.get(key)
        self.__schedule_expiry()
        now = time.time()
        with self.locks[key]:
            if self.storage.get(key, 0) + amount > limit:
                return None
            self.storage[key] += amount
            if self.storage[key] == amount:
                self.expirations[key] = now + expiry
                self.window_generations[key] = self.window_generations.get(key, 0) + 1
            lease = Lease(
                lease_id=uuid.uuid4().hex,
                key=key,
                amount=amount,
                expires_at=now + (lease_ttl if lease_ttl is not None else expiry),
                generation=self.window_generations[key],
            )
            self.leases[lease.lease_id] = LeaseRecord(lease)
            return lease

    def commit(self, lease: Lease, cost: int) -> None:
        """
        settle ``lease`` with the actual ``cost``, releasing the unconsumed
        remainder of the reservation back to the window it was taken from

        :param lease: the lease to settle
        :param cost: the actual cost, which must not exceed the reserved amount
        """
        if cost > lease.amount:
            raise ValueError(f"cost {cost} exceeds the reserved amount {lease.amount}")

        self.__schedule_expiry()
        now = time.time()
        with self.locks[lease.key]:
            record = self.leases.get(lease.lease_id)
            if record is None or record.key != lease.key:
                raise LeaseExpiredError(lease.lease_id)
            self._expire_lease(record, now)
            if record.state == "committed":
                if record.committed_cost == cost:
                    return
                raise LeaseConflictError(lease.lease_id)
            if record.state == "cancelled":
                raise LeaseConflictError(lease.lease_id)
            if record.state == "expired":
                raise LeaseExpiredError(lease.lease_id)
            if not self._window_live(record, now):
                record.state = "expired"
                raise LeaseExpiredError(lease.lease_id)
            self._refund(record, record.amount - cost)
            record.state = "committed"
            record.committed_cost = cost

    def cancel(self, lease: Lease) -> None:
        """
        release the reservation held by ``lease``. A cancel arriving after
        the window moved on or the lease expired does not credit the
        current window.

        :param lease: the lease to release
        """
        self.__schedule_expiry()
        now = time.time()
        with self.locks[lease.key]:
            record = self.leases.get(lease.lease_id)
            if record is None:
                return
            self._expire_lease(record, now)
            if record.state == "open":
                if self._window_live(record, now):
                    self._refund(record, record.amount)
                record.state = "cancelled"
            elif record.state == "committed":
                raise LeaseConflictError(lease.lease_id)


    def acquire_entry(self, key: str, limit: int, expiry: int, amount: int = 1) -> bool:
        """
        :param key: rate limit key to acquire an entry in
        :param limit: amount of entries allowed
        :param expiry: expiry of the entry
        :param amount: the number of entries to acquire
        """
        if amount > limit:
            return False

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
        now = time.time()
        previous_key, current_key = self.sliding_window_keys(key, expiry, now)
        (
            previous_count,
            previous_ttl,
            current_count,
            _,
        ) = self._get_sliding_window_info(previous_key, current_key, expiry, now)
        weighted_count = previous_count * previous_ttl / expiry + current_count
        if floor(weighted_count) + amount > limit:
            return False
        else:
            # Hit, increase the current counter.
            # If the counter doesn't exist yet, set twice the theorical expiry.
            current_count = self.incr(current_key, 2 * expiry, amount=amount)
            weighted_count = previous_count * previous_ttl / expiry + current_count
            if floor(weighted_count) > limit:
                # Another hit won the race condition: revert the incrementation and refuse this hit
                # Limitation: during high concurrency at the end of the window,
                # the counter is shifted and cannot be decremented, so less requests than expected are allowed.
                self.decr(current_key, amount)
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

    def check(self) -> bool:
        """
        check if storage is healthy
        """

        return True

    def reset(self) -> int | None:
        num_items = max(len(self.storage), len(self.events))
        self.storage.clear()
        self.expirations.clear()
        self.events.clear()
        self.locks.clear()
        self.leases.clear()
        self.window_generations.clear()
        return num_items
