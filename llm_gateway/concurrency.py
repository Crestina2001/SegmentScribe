from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from .config import ConcurrencyConfig
from .models import ProviderName


class ConcurrencyController:
    def __init__(self, config: ConcurrencyConfig) -> None:
        self._config = config
        self._global = asyncio.Semaphore(max(1, config.global_limit))
        self._provider_semaphores = {
            provider: asyncio.Semaphore(max(1, limit))
            for provider, limit in config.provider_limits.items()
        }
        self._model_semaphores: dict[str, asyncio.Semaphore] = {}

    @asynccontextmanager
    async def acquire(self, provider: ProviderName, model: str):
        provider_sem = self._provider_semaphores.get(provider)
        model_sem = self._get_model_semaphore(provider=provider, model=model)
        async with self._global:
            if provider_sem is None and model_sem is None:
                yield
                return
            if provider_sem is not None:
                async with provider_sem:
                    if model_sem is not None:
                        async with model_sem:
                            yield
                    else:
                        yield
                return
            async with model_sem:
                yield

    def _get_model_semaphore(self, *, provider: ProviderName, model: str) -> asyncio.Semaphore | None:
        limit = self._config.model_limits.get(f"{provider}:{model}")
        if limit is None:
            limit = self._config.model_limits.get(model)
        if limit is None:
            return None
        key = f"{provider}:{model}"
        if key not in self._model_semaphores:
            self._model_semaphores[key] = asyncio.Semaphore(max(1, limit))
        return self._model_semaphores[key]
