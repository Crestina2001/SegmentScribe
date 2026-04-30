from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..config import ProviderConfig
from ..models import PromptRequest, PromptResponse
from ..retry import ProviderRequestError
from ..tools import ToolCallManager


class BaseProviderAdapter(ABC):
    provider_name: str

    def __init__(self, config: ProviderConfig, *, tool_manager: ToolCallManager) -> None:
        self.config = config
        self.tool_manager = tool_manager

    @abstractmethod
    def prepare_request(self, request: PromptRequest) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    async def send(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def parse_response(self, raw_response: dict[str, Any]) -> PromptResponse:
        raise NotImplementedError

    @abstractmethod
    def is_retryable_error(self, error: Exception) -> bool:
        raise NotImplementedError

    async def close(self) -> None:
        return None

    def _coerce_error(self, error: Exception) -> ProviderRequestError:
        if isinstance(error, ProviderRequestError):
            return error
        return ProviderRequestError(str(error), retryable=self.is_retryable_error(error), raw_error=error)
