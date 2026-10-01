"""
errors and exceptions
"""

from __future__ import annotations


class ConfigurationError(Exception):
    """
    Error raised when a configuration problem is encountered
    """


class ConcurrentUpdateError(Exception):
    """
    Error raised when an update to limit fails due to concurrent
    updates
    """

    def __init__(self, key: str, attempts: int) -> None:
        super().__init__(f"Unable to update {key} after {attempts} retries")


class StorageError(Exception):
    """
    Error raised when an error is encountered in a storage
    """

    def __init__(self, storage_error: Exception) -> None:
        self.storage_error = storage_error


class ReservationError(Exception):
    """
    Base error for failures while settling a reservation lease
    """


class LeaseExpiredError(ReservationError):
    """
    Error raised when settling a lease that is no longer valid:
    the lease expired, its window was reset, or it was already reclaimed
    """

    def __init__(self, lease_id: str) -> None:
        super().__init__(f"Lease {lease_id} is expired or no longer valid")


class LeaseConflictError(ReservationError):
    """
    Error raised when a lease is settled with an outcome that
    conflicts with a previous settlement of the same lease
    """

    def __init__(self, lease_id: str) -> None:
        super().__init__(
            f"Lease {lease_id} was already settled with a different outcome"
        )
