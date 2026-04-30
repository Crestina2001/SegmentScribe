from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

from .config import RetryConfig

T = TypeVar("T")


@dataclass(slots=True)
class ProviderRequestError(Exception):
    message: str
    status_code: int | None = None
    retryable: bool = False
    raw_error: Exception | None = None

    def __str__(self) -> str:
        return self.message


async def execute_with_retry(
    operation: Callable[[], Awaitable[T]],
    *,
    config: RetryConfig,
    should_retry: Callable[[Exception], bool],
) -> T:
    attempts = max(1, config.max_retries + 1)
    for attempt in range(1, attempts + 1):
        try:
            return await operation()
        except Exception as exc:
            if attempt >= attempts or not should_retry(exc):
                raise
            await asyncio.sleep(_compute_delay(config=config, attempt=attempt))
    raise RuntimeError("Retry loop exited unexpectedly.")


def _compute_delay(*, config: RetryConfig, attempt: int) -> float:
    base = config.base_delay_seconds * (2 ** (attempt - 1))
    capped = min(base, config.max_delay_seconds)
    jitter = random.uniform(0.0, max(0.0, config.jitter_seconds))
    return capped + jitter
