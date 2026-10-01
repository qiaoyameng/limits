from __future__ import annotations

import time
from dataclasses import dataclass

from limits.limits import RateLimitItem
from limits.typing import Literal

CombinedWindowStrategy = Literal[
    "fixed-window",
    "sliding-window-counter",
    "moving-window",
]


@dataclass(frozen=True)
class CombinedRateLimitRequest:
    item: RateLimitItem
    identifiers: tuple[str, ...] = ()
    cost: int = 1
    strategy: CombinedWindowStrategy = "fixed-window"
    source: int | str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.item, RateLimitItem):
            raise ValueError("rate limit item is required")
        if type(self.cost) is not int or self.cost <= 0:
            raise ValueError("rate limit cost must be a positive integer")
        if self.strategy not in {
            "fixed-window",
            "sliding-window-counter",
            "moving-window",
        }:
            raise ValueError(f"unsupported rate limit strategy: {self.strategy}")

    @classmethod
    def fixed(
        cls,
        item: RateLimitItem,
        *identifiers: object,
        cost: int = 1,
        source: int | str | None = None,
    ) -> CombinedRateLimitRequest:
        return cls(
            item,
            tuple(str(identifier) for identifier in identifiers),
            cost,
            "fixed-window",
            source,
        )

    @classmethod
    def sliding(
        cls,
        item: RateLimitItem,
        *identifiers: object,
        cost: int = 1,
        source: int | str | None = None,
    ) -> CombinedRateLimitRequest:
        return cls(
            item,
            tuple(str(identifier) for identifier in identifiers),
            cost,
            "sliding-window-counter",
            source,
        )


@dataclass(frozen=True)
class CombinedRateLimitRejection:
    source: int | str
    key: str
    retry_at: float | None

    @property
    def retry_in(self) -> float | None:
        if self.retry_at is None:
            return None
        return max(0.0, self.retry_at - time.time())


@dataclass(frozen=True)
class CombinedRateLimitResult:
    allowed: bool
    rejected: tuple[CombinedRateLimitRejection, ...] = ()
    retry_at: float | None = None

    @property
    def retry_in(self) -> float | None:
        if self.retry_at is None:
            return None
        return max(0.0, self.retry_at - time.time())

    def __bool__(self) -> bool:
        return self.allowed
