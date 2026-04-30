from __future__ import annotations

import re

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


class MiniMaxAdapter(BaseProviderAdapter):
    provider_name = "minimax"

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
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.response_format is not None:
            payload["response_format"] = request.response_format
        if request.tools:
            payload["tools"] = request.tools
        if request.tool_choice is not None:
            payload["tool_choice"] = request.tool_choice
        if _uses_reasoning_split(request.model):
            payload["extra_body"] = {"reasoning_split": True}
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
        text = _extract_visible_text(message.get("content"))
        reasoning = _extract_reasoning(message)

        return PromptResponse(
            text=text,
            tool_calls=self.tool_manager.parse_tool_calls(message),
            finish_reason=first_choice.get("finish_reason"),
            provider="minimax",
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


_THINK_BLOCK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL | re.IGNORECASE)


def _uses_reasoning_split(model: str) -> bool:
    return model.strip().lower().startswith("minimax-m2.7")


def _extract_visible_text(content: object) -> str:
    if isinstance(content, list):
        text = "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") in {"text", "output_text"}
        )
        return text.strip()

    raw_text = str(content or "")
    return _THINK_BLOCK_RE.sub("", raw_text).strip()


def _extract_reasoning(message: dict[str, object]) -> str | None:
    reasoning_details = message.get("reasoning_details")
    if isinstance(reasoning_details, list):
        parts = [
            str(item.get("text") or "").strip()
            for item in reasoning_details
            if isinstance(item, dict) and str(item.get("text") or "").strip()
        ]
        if parts:
            return "\n\n".join(parts)

    content = message.get("content")
    if isinstance(content, str):
        matches = _THINK_BLOCK_RE.findall(content)
        parts = [match.strip() for match in matches if match.strip()]
        if parts:
            return "\n\n".join(parts)
    return None
