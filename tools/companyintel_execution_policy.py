"""Code-owned CompanyIntel execution policy primitives.

This module is transport-independent: workers use it to normalize outcomes and
admit work before mutating the durable graph. It never performs network I/O.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from time import monotonic


class SourceOutcome(StrEnum):
    SUCCESS = "SUCCESS"
    NO_RESULTS = "NO_RESULTS"
    RETRY_WAIT = "RETRY_WAIT"
    RESUMABLE = "RESUMABLE"
    BLOCKED = "BLOCKED"
    RATE_LIMITED = "RATE_LIMITED"
    UNSUPPORTED = "UNSUPPORTED"
    PARTIAL = "PARTIAL"
    TERMINAL_ERROR = "TERMINAL_ERROR"


@dataclass(frozen=True)
class RateLimitPolicy:
    per_source: int = 10
    per_domain: int = 20
    window_seconds: float = 60.0

    def __post_init__(self) -> None:
        if self.per_source < 1 or self.per_domain < 1 or self.window_seconds <= 0:
            raise ValueError("rate-limit values must be positive")

    def admit(self, *, source_calls: int, domain_calls: int) -> bool:
        return source_calls < self.per_source and domain_calls < self.per_domain


@dataclass(frozen=True)
class ExecutionBudget:
    max_tasks: int
    max_searches: int
    deadline_at: float | None = None

    def admit(self, *, tasks_executed: int, source_calls: int, now: float | None = None) -> bool:
        current = monotonic() if now is None else now
        return (
            tasks_executed < self.max_tasks
            and source_calls < self.max_searches
            and (self.deadline_at is None or current < self.deadline_at)
        )


def normalize_source_outcome(value: str, *, retryable: bool = False, has_data: bool = False) -> SourceOutcome:
    raw = "_".join(str(value or "").strip().upper().split())
    if raw in SourceOutcome.__members__:
        return SourceOutcome[raw]
    if retryable:
        return SourceOutcome.RETRY_WAIT
    if has_data:
        return SourceOutcome.PARTIAL
    return SourceOutcome.TERMINAL_ERROR
