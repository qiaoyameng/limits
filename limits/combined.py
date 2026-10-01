from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Literal, Sequence

from limits.errors import ConfigurationError
from limits.limits import RateLimitItem
from limits.storage.base import TimestampedSlidingWindow
from limits.typing import Any

CombinedWindow = Literal["fixed-window", "sliding-window-counter"]


@dataclass(frozen=True, slots=True)
class CombinedRateLimitRequest:
    """A single rate limit requirement in an atomic group."""

    item: RateLimitItem
    identifiers: tuple[Any, ...]
    cost: int
    window: CombinedWindow
    explicit_key: str | None = None

    @classmethod
    def fixed_window(
        cls,
        item: RateLimitItem,
        *identifiers: Any,
        key: str | None = None,
        cost: int = 1,
    ) -> CombinedRateLimitRequest:
        return cls(
            item, identifiers, cost, "fixed-window", explicit_key=key
        )

    @classmethod
    def sliding_window_counter(
        cls,
        item: RateLimitItem,
        *identifiers: Any,
        key: str | None = None,
        cost: int = 1,
    ) -> CombinedRateLimitRequest:
        return cls(item, identifiers, cost, "sliding-window-counter", explicit_key=key)

    @property
    def key(self) -> str:
        return self.explicit_key or self.item.key_for(*self.identifiers)


@dataclass(frozen=True, slots=True)
class CombinedRateLimitResult:
    """The result of attempting to consume an atomic group."""

    allowed: bool
    rejected: tuple[CombinedRateLimitRequest, ...]
    retry_at: float

    def __bool__(self) -> bool:
        return self.allowed


@dataclass(slots=True)
class _CombinedWindowGroup:
    window: CombinedWindow
    key: str
    limit: int
    expiry: int
    cost: int
    sources: list[int]
    physical_signature: tuple[str, ...]


def validate_combined_requests(
    requests: Sequence[CombinedRateLimitRequest],
) -> tuple[CombinedRateLimitRequest, ...]:
    if not isinstance(requests, (list, tuple)):
        raise TypeError("combined rate limit requests must be a list or tuple")
    if len(requests) == 0:
        raise ValueError("combined rate limit requests must not be empty")

    validated: list[CombinedRateLimitRequest] = []
    for request in requests:
        if not isinstance(request, CombinedRateLimitRequest):
            raise TypeError(
                "each combined rate limit request must be a CombinedRateLimitRequest"
            )
        if not isinstance(request.item, RateLimitItem):
            raise TypeError("combined rate limit item must be a RateLimitItem")
        if request.window not in ("fixed-window", "sliding-window-counter"):
            raise ValueError(
                f"unsupported combined rate limit window: {request.window}"
            )
        if isinstance(request.cost, bool) or not isinstance(request.cost, int):
            raise TypeError("combined rate limit cost must be a positive integer")
        if request.cost <= 0:
            raise ValueError("combined rate limit cost must be a positive integer")
        validated.append(request)

    return tuple(validated)


def prepare_combined_requests(
    requests: Sequence[CombinedRateLimitRequest],
    now: float,
) -> tuple[_CombinedWindowGroup, ...]:
    validated = validate_combined_requests(requests)
    groups: dict[tuple[CombinedWindow, str], _CombinedWindowGroup] = {}
    group_ids: dict[tuple[CombinedWindow, str], int] = {}
    physical_keys: dict[str, int] = {}

    for source, request in enumerate(validated):
        logical_key = (
            request.window,
            request.key,
            request.item.amount,
            request.item.get_expiry(),
        )
        group = groups.get(logical_key)
        if group is None:
            physical_signature = (request.key,)
            if request.window == "sliding-window-counter":
                physical_signature = TimestampedSlidingWindow.sliding_window_keys(
                    request.key, request.item.get_expiry(), now
                )
            group = _CombinedWindowGroup(
                request.window,
                request.key,
                request.item.amount,
                request.item.get_expiry(),
                0,
                [],
                physical_signature,
            )
            groups[logical_key] = group
            group_id = len(groups) - 1
            group_ids[logical_key] = group_id

        group.cost += request.cost
        group.sources.append(source)

        for physical_key in group.physical_signature:
            existing_group_id = physical_keys.get(physical_key)
            group_id = group_ids[logical_key]
            if existing_group_id is not None and existing_group_id != group_id:
                raise ConfigurationError(
                    "combined rate limit requests share physical storage key "
                    f"{physical_key}"
                )
            physical_keys[physical_key] = group_id

    return tuple(groups.values())


def fixed_window_retry_at(
    current_count: int,
    cost: int,
    limit: int,
    expiry: float,
) -> float:
    if cost > limit:
        return inf
    return expiry


def sliding_window_retry_at(
    now: float,
    previous_count: int,
    previous_ttl: float,
    current_count: int,
    cost: int,
    limit: int,
    expiry: int,
) -> float:
    if cost > limit:
        return inf

    if current_count + cost <= limit:
        if previous_count == 0:
            return inf
        threshold = limit - cost - current_count + 1
        elapsed = previous_ttl - threshold * expiry / previous_count
        return now + max(0.0, elapsed)

    current_boundary = (int(now // expiry) + 1) * expiry
    if current_count == 0:
        return inf

    threshold = limit - cost + 1
    return current_boundary + expiry - threshold * expiry / current_count
