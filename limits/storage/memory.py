from __future__ import annotations

import bisect
import threading
import time
import uuid
from collections import Counter, defaultdict
from math import floor

import limits.typing
from limits import errors
from limits.storage.base import (
    Lease,
    MovingWindowSupport,
    ReservationSupport,
    SlidingWindowCounterSupport,
    Storage,
    TimestampedSlidingWindow,
)


class Entry:
    def __init__(self, expiry: float) -> None:
        self.atime = time.time()
        self.expiry = self.atime + expiry


class LeaseRecord:
    """
    Internal bookkeeping for a reservation lease held by the storage.
    """

    def __init__(self, lease: Lease) -> None:
        self.lease = lease
        self.state = "open"
        self.settled_cost: int | None = None


class MemoryStorage(
    Storage,
    MovingWindowSupport,
    SlidingWindowCounterSupport,
    TimestampedSlidingWindow,
    ReservationSupport,
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
            if record and record.lease.expires_at <= time.time():
                with self.locks[record.lease.key]:
                    self.__release_lease(record)
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
        for lease_id, record in list(self.leases.items()):
            if record.lease.key == key:
                self.leases.pop(lease_id, None)

    def __release_lease(self, record: LeaseRecord) -> None:
        """
        releases the unconsumed reservation of an open lease back to
        the window it was reserved from, provided that window is still
        current (i.e. the window generation matches).
        """
        lease = record.lease
        if record.state == "open" and self.expirations.get(lease.key) == (
            lease.generation
        ):
            self.storage[lease.key] = max(
                self.storage.get(lease.key, 0) - lease.amount, 0
            )

    def reserve(
        self,
        key: str,
        limit: int,
        expiry: float,
        amount: int = 1,
        lease_expiry: float | None = None,
    ) -> Lease | None:
        """
        :param key: rate limit key to reserve from
        :param limit: amount of entries allowed
        :param expiry: expiry of the window the reservation belongs to
        :param amount: the (upper bound) amount to reserve
        :param lease_expiry: seconds until the lease itself expires and
         is released. Defaults to ``expiry``.
        :return: the lease, or ``None`` if the reservation would exceed
         the limit
        """
        if amount < 1:
            raise ValueError("amount must be at least 1")
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
            lease = Lease(
                lease_id=uuid.uuid4().hex,
                key=key,
                generation=self.expirations[key],
                amount=amount,
                expires_at=now + (expiry if lease_expiry is None else lease_expiry),
            )
            self.leases[lease.lease_id] = LeaseRecord(lease)
            return lease

    def commit(self, lease: Lease, cost: int) -> None:
        """
        settles ``lease`` with the actual ``cost`` and releases the
        unconsumed remainder of the reservation

        :param lease: the lease returned by :meth:`reserve`
        :param cost: the actual cost, must not exceed the reserved amount
        """
        if cost < 0:
            raise ValueError(f"cost {cost} must not be negative")
        if cost > lease.amount:
            raise ValueError(
                f"cost {cost} exceeds the reserved amount {lease.amount}"
            )

        self.__schedule_expiry()
        with self.locks[lease.key]:
            record = self.leases.get(lease.lease_id)
            if record is None:
                raise errors.LeaseExpiredError(lease.lease_id)
            if record.state == "open" and record.lease.expires_at <= time.time():
                self.__release_lease(record)
                self.leases.pop(lease.lease_id, None)
                raise errors.LeaseExpiredError(lease.lease_id)
            if record.state == "committed":
                if record.settled_cost == cost:
                    return
                raise errors.LeaseConflictError(lease.lease_id)
            if record.state == "cancelled":
                raise errors.LeaseConflictError(lease.lease_id)

            refund = record.lease.amount - cost
            if refund and self.expirations.get(record.lease.key) == (
                record.lease.generation
            ):
                self.storage[record.lease.key] = max(
                    self.storage.get(record.lease.key, 0) - refund, 0
                )
            record.state = "committed"
            record.settled_cost = cost

    def cancel(self, lease: Lease) -> None:
        """
        releases the full reservation held by ``lease``

        :param lease: the lease returned by :meth:`reserve`
        """
        self.__schedule_expiry()
        with self.locks[lease.key]:
            record = self.leases.get(lease.lease_id)
            if record is None:
                raise errors.LeaseExpiredError(lease.lease_id)
            if record.state == "open" and record.lease.expires_at <= time.time():
                self.__release_lease(record)
                self.leases.pop(lease.lease_id, None)
                raise errors.LeaseExpiredError(lease.lease_id)
            if record.state == "cancelled":
                return
            if record.state == "committed":
                raise errors.LeaseConflictError(lease.lease_id)

            self.__release_lease(record)
            record.state = "cancelled"

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
        return num_items
