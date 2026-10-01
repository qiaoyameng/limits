from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class Clock(Protocol):
    def time(self) -> float:
        ...


@dataclass(slots=True)
class ReferenceEntry:
    atime: float
    expiry: float


class ReferenceMovingWindow:
    def __init__(self, clock: Clock) -> None:
        self.clock = clock
        self.events: list[ReferenceEntry] = []

    def expire(self) -> None:
        now = self.clock.time()
        self.events = [entry for entry in self.events if entry.expiry > now]

    def acquire(self, limit: int, expiry: float, amount: int = 1) -> bool:
        if amount > limit:
            return False
        timestamp = self.clock.time()
        try:
            entry = self.events[limit - amount]
        except IndexError:
            entry = None
        if entry and entry.atime >= timestamp - expiry:
            return False
        self.events[:0] = [
            ReferenceEntry(timestamp, timestamp + expiry) for _ in range(amount)
        ]
        return True

    def get_moving_window(self, expiry: float) -> tuple[float, int]:
        timestamp = self.clock.time()
        if self.events:
            return self.events[-1].atime, sum(
                1 for entry in self.events if entry.atime > timestamp - expiry
            )
        return timestamp, 0
