"""
Generic Cell Rate Algorithm (GCRA) primitives.

GCRA keeps track of a single theoretical arrival time (:math:`T_{at}`) per
rate limit key instead of a historical entry per request. The state is
advanced only when a request is accepted and a key whose theoretical arrival
time is already in the past behaves exactly like a key that was never seen.

To keep the schedule exactly on the emission grid regardless of how many
requests have been processed, the state is stored as ``(base, units)`` where
``base`` is the anchor time of the currently queued schedule and ``units`` is
the integer number of emission intervals that have been scheduled from it.
The theoretical arrival time is ``base + units * emission_interval``. When
the schedule fully elapses the state expires and the next request starts a
new schedule anchored at its own arrival time.

Reference: https://en.wikipedia.org/wiki/Generic_cell_rate_algorithm
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from .limits import RateLimitItem

#: GCRA per-key state: ``(base, units)``
GCRState = tuple[float, int]


@dataclass(frozen=True, slots=True)
class GCRAParameters:
    """
    Public definition of a rate limit expressed in GCRA terms.

    :param amount: the average rate, i.e. the number of requests permitted
     per :attr:`period`.
    :param period: the duration, in seconds, over which :attr:`amount`
     requests are permitted on average.
    :param burst: the number of requests (in units of ``cost``) that may be
     accommodated in a single burst. Defaults to :attr:`amount`.
    """

    amount: int
    period: float
    burst: int | None = None

    def __post_init__(self) -> None:
        if self.amount <= 0:
            raise ValueError("average rate amount must be a positive integer")
        if self.period <= 0:
            raise ValueError("average rate period must be greater than 0")
        if self.burst is not None and self.burst <= 0:
            raise ValueError("allowed burst must be a positive integer")

    @classmethod
    def from_rate_limit(cls, item: RateLimitItem) -> GCRAParameters:
        """
        Build the GCRA parameters for a :class:`~limits.limits.RateLimitItem`.
        The allowed burst defaults to the rate limit amount.
        """
        return cls(item.amount, item.get_expiry())

    @property
    def burst_amount(self) -> int:
        """the number of requests that may be accommodated in a single burst"""
        return self.amount if self.burst is None else self.burst

    @property
    def emission_interval(self) -> float:
        """
        the time, in seconds, that has to separate two requests of unit cost
        on average.
        """
        return self.period / self.amount

    @property
    def burst_units(self) -> int:
        """the number of emission intervals of accumulated idle tolerance"""
        return self.burst_amount - 1

    @property
    def burst_limit(self) -> float:
        """the tolerance, in seconds, accumulated during idle time"""
        return self.emission_interval * self.burst_units

    @property
    def average_rate(self) -> float:
        """the sustained rate in requests per second"""
        return self.amount / self.period


class GCRAAllowance(NamedTuple):
    """
    Result of evaluating a single request against the stored GCRA state.
    """

    #: whether the request is allowed
    allowed: bool
    #: the new state to persist (the old state when rejected)
    state: GCRState
    #: earliest time at which a request of the same cost would be allowed
    retry_after: float


def theoretical_arrival(
    now: float, state: GCRState | None, emission_interval: float
) -> float:
    """return the effective theoretical arrival time of ``state`` at ``now``"""
    if state is None:
        return now
    base, units = state
    return max(base + units * emission_interval, now)


def evaluate(
    now: float,
    state: GCRState | None,
    emission_interval: float,
    burst_limit: float,
    cost: int = 1,
) -> GCRAAllowance:
    """
    Decide whether a request of ``cost`` arriving at ``now`` may be accepted.

    The state is only advanced when the request is accepted. A single
    ``cost`` larger than the accommodated burst is rejected immediately,
    even when the limit is completely idle.

    :param now: the current time in seconds
    :param state: the stored ``(base, units)`` state, or ``None``
    :param emission_interval: the time a unit of ``cost`` occupies
    :param burst_limit: the accumulated idle-time tolerance
    :param cost: the weight of the request, default 1
    """
    if cost <= 0:
        raise ValueError("cost must be a positive integer")

    burst_units = round(burst_limit / emission_interval)
    if cost - 1 > burst_units:
        # The request can never fit, no matter how long the key is idle.
        return GCRAAllowance(False, state or (now, 0), float("inf"))

    if state is not None:
        base, units = state
        if base + units * emission_interval <= now:
            state = None

    if state is None:
        return GCRAAllowance(
            True,
            (now, cost),
            now + (burst_units + 1) * emission_interval,
        )

    base, units = state
    if units + cost - 1 > burst_units:
        return GCRAAllowance(
            False,
            state,
            base + (units + cost - 1 - burst_units) * emission_interval,
        )

    return GCRAAllowance(
        True,
        (base, units + cost),
        base + (units + 2 * cost - 1 - burst_units) * emission_interval,
    )


def remaining(
    state: GCRState | None, emission_interval: float, burst_amount: int
) -> int:
    """
    Return the number of unit-cost requests accepted immediately at the
    state's current observation time.
    """
    if state is None:
        return burst_amount
    _, units = state
    return max(0, burst_amount - units)


def reset_time(
    now: float,
    state: GCRState | None,
    emission_interval: float,
    burst_amount: int,
) -> float:
    """
    Return the earliest time at which a unit-cost request would be accepted.
    This is the current time while the limit has any remaining capacity and
    never a fixed-window style reset time.
    """
    if state is None:
        return now
    base, units = state
    if base + units * emission_interval <= now:
        return now
    if units < burst_amount:
        return now
    return base + (units - burst_amount + 1) * emission_interval
