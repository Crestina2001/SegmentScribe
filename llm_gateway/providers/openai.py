from __future__ import annotations

import warnings

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


class OpenAIAdapter(BaseProviderAdapter):
    provider_name = "openai"

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
        if request.reasoning_effort is not None:
            if _supports_reasoning_effort(request.model):
                payload["reasoning_effort"] = request.reasoning_effort
            else:
                warnings.warn(
                    f"OpenAI model {request.model!r} does not support reasoning_effort; omitting it from the request.",
                    stacklevel=2,
                )
        if request.temperature is not None and not _uses_reasoning_defaults(request.model):
            payload["temperature"] = request.temperature
        if request.top_p is not None and not _uses_reasoning_defaults(request.model):
            payload["top_p"] = request.top_p
        if request.max_tokens is not None:
            if _uses_max_completion_tokens(request.model):
                payload["max_completion_tokens"] = request.max_tokens
            else:
                payload["max_tokens"] = request.max_tokens
        if request.response_format is not None:
            payload["response_format"] = request.response_format
        if request.tools:
            payload["tools"] = request.tools
        if request.tool_choice is not None:
            payload["tool_choice"] = request.tool_choice
        return payload

    async def send(self, payload: dict[str, object]) -> dict[str, object]:
        try:
            response = await self._client.chat.completions.create(**payload)
        except Exception as exc:
            raise self._wrap_error(exc) from exc
        return self._response_to_dict(response)

    @staticmethod
    def _response_to_dict(response: object) -> dict[str, object]:
        if isinstance(response, dict):
            return response

        if isinstance(response, str):
            preview = response[:200].replace("\n", "\\n")
            raise ProviderRequestError(
                message=f"OpenAI provider returned raw string response instead of chat completion object: {preview!r}",
                retryable=True,
                raw_error=RuntimeError(response),
            )

        model_dump = getattr(response, "model_dump", None)
        if callable(model_dump):
            dumped = model_dump()
            if isinstance(dumped, dict):
                return dumped
            raise ProviderRequestError(
                message=(
                    "OpenAI provider returned a chat completion object whose model_dump() "
                    f"produced unsupported type: {type(dumped).__name__}"
                ),
                retryable=True,
                raw_error=RuntimeError(repr(dumped)),
            )

        raise ProviderRequestError(
            message=f"OpenAI provider returned unsupported response type: {type(response).__name__}",
            retryable=True,
            raw_error=RuntimeError(repr(response)),
        )

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

        return PromptResponse(
            text=text,
            tool_calls=self.tool_manager.parse_tool_calls(message),
            finish_reason=first_choice.get("finish_reason"),
            provider="openai",
            model=raw_response.get("model"),
            usage=raw_response.get("usage"),
            raw=raw_response,
            reasoning=message.get("reasoning_content"),
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
    lowered = model.strip().lower()
    return "gpt-5" in lowered or lowered.startswith(("o1", "o3", "o4"))


def _uses_max_completion_tokens(model: str) -> bool:
    lowered = model.strip().lower()
    return "gpt-5" in lowered or lowered.startswith(("o1", "o3", "o4"))


def _supports_reasoning_effort(model: str) -> bool:
    return "gpt-5" in model.strip().lower()
