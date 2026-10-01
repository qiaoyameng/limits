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


class LeaseError(Exception):
    """
    Base error for reservation lease operations
    """


class LeaseConflictError(LeaseError):
    """
    Error raised when a lease is settled with a conflicting
    outcome (e.g. committed twice with different costs, or
    cancelled after being committed)
    """

    def __init__(self, lease_id: str) -> None:
        super().__init__(f"Conflicting settlement for lease {lease_id}")


class LeaseExpiredError(LeaseError):
    """
    Error raised when operating on a lease that is unknown,
    expired or has already been reclaimed
    """

    def __init__(self, lease_id: str) -> None:
        super().__init__(f"Lease {lease_id} is expired or unknown")
