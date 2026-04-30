from __future__ import annotations

import httpx
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)

from ..config import ProviderConfig
from ..models import PromptRequest, PromptResponse
from ..retry import ProviderRequestError
from ..tools import ToolCallManager
from .base import BaseProviderAdapter


class DeepSeekAdapter(BaseProviderAdapter):
    """DeepSeek via its OpenAI-compatible chat completions endpoint."""

    provider_name = "deepseek"

    def __init__(self, config: ProviderConfig, *, tool_manager: ToolCallManager) -> None:
        super().__init__(config, tool_manager=tool_manager)
        self._http_client = self._build_http_client(config)
        self._client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.request_timeout,
            http_client=self._http_client,
        )

    def prepare_request(self, request: PromptRequest) -> dict[str, object]:
        payload: dict[str, object] = {"model": request.model, "messages": request.messages}
        if request.temperature is not None and not _uses_reasoning_defaults(request.model):
            payload["temperature"] = request.temperature
        if request.top_p is not None and not _uses_reasoning_defaults(request.model):
            payload["top_p"] = request.top_p
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.response_format is not None:
            payload["response_format"] = request.response_format
        if request.tools:
            payload["tools"] = request.tools
            payload["extra_body"] = {"thinking": {"type": "disabled"}}
        if request.tool_choice is not None and not _drops_tool_choice_required(request.tool_choice):
            payload["tool_choice"] = request.tool_choice
        return payload

    async def send(self, payload: dict[str, object]) -> dict[str, object]:
        try:
            response = await self._client.chat.completions.create(**payload)
        except Exception as exc:
            raise self._wrap_error(exc) from exc
        return response.model_dump()

    def parse_response(self, raw_response: dict[str, object]) -> PromptResponse:
        choices = raw_response.get("choices") or []
        first_choice = choices[0] if choices else {}
        message = first_choice.get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            text = "".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and item.get("type") in {"text", "output_text"}
            )
        else:
            text = str(content or "")

        reasoning = message.get("reasoning_content")
        if reasoning is not None:
            reasoning = str(reasoning)

        return PromptResponse(
            text=text,
            tool_calls=self.tool_manager.parse_tool_calls(message),
            finish_reason=first_choice.get("finish_reason"),
            provider="deepseek",
            model=raw_response.get("model"),
            usage=raw_response.get("usage"),
            raw=raw_response,
            reasoning=reasoning,
            response_id=raw_response.get("id"),
        )

    def is_retryable_error(self, error: Exception) -> bool:
        if isinstance(error, ProviderRequestError):
            return error.retryable
        if isinstance(error, (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)):
            return True
        if isinstance(error, APIStatusError):
            return error.status_code in {408, 409, 429} or error.status_code >= 500
        status_code = getattr(error, "status_code", None)
        return isinstance(status_code, int) and (status_code in {408, 409, 429} or status_code >= 500)

    async def close(self) -> None:
        await self._http_client.aclose()

    def _wrap_error(self, error: Exception) -> ProviderRequestError:
        status_code = getattr(error, "status_code", None)
        if isinstance(error, APIStatusError):
            status_code = error.status_code
        return ProviderRequestError(
            message=str(error),
            status_code=status_code if isinstance(status_code, int) else None,
            retryable=self.is_retryable_error(error),
            raw_error=error,
        )

    @staticmethod
    def _build_http_client(config: ProviderConfig) -> httpx.AsyncClient:
        proxy = config.network_proxy
        if not proxy:
            return httpx.AsyncClient(timeout=config.request_timeout)
        return httpx.AsyncClient(timeout=config.request_timeout, proxy=proxy)


def _uses_reasoning_defaults(model: str) -> bool:
    return model.strip().lower() == "deepseek-reasoner"


def _drops_tool_choice_required(tool_choice: str | object) -> bool:
    return isinstance(tool_choice, str) and tool_choice.strip().lower() == "required"
