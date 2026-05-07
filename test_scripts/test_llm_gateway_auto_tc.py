from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from llm_gateway import MemoryManager, Tool, ToolCallManager, UnifiedClient
from llm_gateway.config import GatewayConfig, ProviderConfig, RetryConfig
from llm_gateway.models import PromptRequest, PromptResponse
from llm_gateway.providers.base import BaseProviderAdapter
from llm_gateway.providers.openai import OpenAIAdapter
from llm_gateway.retry import ProviderRequestError
from slide_LLM.pipeline import PooledLLMClient


class FakeAdapter(BaseProviderAdapter):
    provider_name = "openai"

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        super().__init__(ProviderConfig(api_key="test"), tool_manager=ToolCallManager())
        self.responses = list(responses)
        self.payloads: list[dict[str, Any]] = []

    def prepare_request(self, request: PromptRequest) -> dict[str, Any]:
        return {
            "model": request.model,
            "messages": request.messages,
            "tools": request.tools,
            "tool_choice": request.tool_choice,
        }

    async def send(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.payloads.append(payload)
        if not self.responses:
            raise AssertionError("No fake response queued.")
        return self.responses.pop(0)

    def parse_response(self, raw_response: dict[str, Any]) -> PromptResponse:
        choice = (raw_response.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        return PromptResponse(
            text=str(message.get("content") or ""),
            tool_calls=self.tool_manager.parse_tool_calls(message),
            finish_reason=choice.get("finish_reason"),
            provider="openai",
            model=raw_response.get("model"),
            raw=raw_response,
            response_id=raw_response.get("id"),
        )

    def is_retryable_error(self, error: Exception) -> bool:
        return False


class FakeClient(UnifiedClient):
    def __init__(self, adapter: FakeAdapter) -> None:
        super().__init__(
            config=GatewayConfig(
                openai=ProviderConfig(api_key="test", models=("test-model",)),
                minimax=ProviderConfig(api_key="test"),
                retry=RetryConfig(max_retries=0),
            )
        )
        self.adapter = adapter

    def _get_adapter(self, provider):  # type: ignore[override]
        return self.adapter


def _tool_call_response(*, call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "resp-tool",
        "model": "test-model",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ],
                },
            }
        ],
    }


def _final_response(text: str = "done") -> dict[str, Any]:
    return {
        "id": "resp-final",
        "model": "test-model",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": text},
            }
        ],
    }


def test_tool_normalizes_to_openai_tool_definition() -> None:
    tool = Tool(
        name="get_weather",
        description="Get weather.",
        parameters={"type": "object", "properties": {"city": {"type": "string"}}},
        func=lambda city: {"city": city},
    )

    normalized = ToolCallManager().normalize_definitions([tool])

    assert normalized == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather.",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]


def test_openai_adapter_marks_raw_string_response_retryable() -> None:
    with pytest.raises(ProviderRequestError) as exc_info:
        OpenAIAdapter._response_to_dict("temporary proxy error")

    assert exc_info.value.retryable is True
    assert "raw string response" in str(exc_info.value)


@pytest.mark.asyncio
async def test_send_prompt_autoTC_executes_sync_tool_and_returns_final_response() -> None:
    adapter = FakeAdapter(
        [
            _tool_call_response(call_id="call_123", name="get_weather", arguments={"city": "Tokyo"}),
            _final_response("Tokyo is clear."),
        ]
    )
    client = FakeClient(adapter)
    tool = Tool(
        name="get_weather",
        description="Get weather.",
        parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        func=lambda city: {"city": city, "temperature": 25},
    )

    response = await client.send_prompt_autoTC(
        MemoryManager(),
        "weather?",
        provider="openai",
        model="test-model",
        tools=[tool],
    )

    assert response.text == "Tokyo is clear."
    assert len(adapter.payloads) == 2
    tool_message = adapter.payloads[1]["messages"][-1]
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "call_123"
    assert json.loads(tool_message["content"]) == {"city": "Tokyo", "temperature": 25}


@pytest.mark.asyncio
async def test_send_prompt_autoTC_executes_async_tool_and_custom_serializer() -> None:
    async def get_weather(city: str) -> dict[str, Any]:
        return {"city": city, "temperature": 25}

    adapter = FakeAdapter(
        [
            _tool_call_response(call_id="call_async", name="get_weather", arguments={"city": "Tokyo"}),
            _final_response(),
        ]
    )
    client = FakeClient(adapter)
    tool = Tool(
        name="get_weather",
        description="Get weather.",
        parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        func=get_weather,
        serialize=lambda result: f"{result['city']}={result['temperature']}",
    )

    await client.send_prompt_autoTC(MemoryManager(), "weather?", provider="openai", model="test-model", tools=[tool])

    assert adapter.payloads[1]["messages"][-1]["content"] == "Tokyo=25"


@pytest.mark.asyncio
async def test_send_prompt_autoTC_preserves_multiple_tool_call_ids() -> None:
    response = _tool_call_response(call_id="call_a", name="echo", arguments={"value": "a"})
    response["choices"][0]["message"]["tool_calls"].append(
        {
            "id": "call_b",
            "type": "function",
            "function": {"name": "echo", "arguments": json.dumps({"value": "b"})},
        }
    )
    adapter = FakeAdapter([response, _final_response()])
    client = FakeClient(adapter)
    tool = Tool(
        name="echo",
        description="Echo value.",
        parameters={"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
        func=lambda value: value,
    )

    await client.send_prompt_autoTC(MemoryManager(), "echo", provider="openai", model="test-model", tools=[tool])

    assert [message["tool_call_id"] for message in adapter.payloads[1]["messages"][-2:]] == ["call_a", "call_b"]
    assert [message["content"] for message in adapter.payloads[1]["messages"][-2:]] == ["a", "b"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool", "expected_error"),
    [
        (
            Tool(
                name="known",
                description="Known.",
                parameters={"type": "object", "properties": {}},
                func=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
            ),
            "boom",
        ),
        (
            Tool(
                name="known",
                description="Known.",
                parameters={"type": "object", "properties": {}},
                func=lambda: {"ok": True},
                serialize=lambda _result: (_ for _ in ()).throw(ValueError("bad serialize")),
            ),
            "bad serialize",
        ),
    ],
)
async def test_send_prompt_autoTC_returns_structured_tool_errors(tool: Tool, expected_error: str) -> None:
    adapter = FakeAdapter(
        [
            _tool_call_response(call_id="call_err", name="known", arguments={}),
            _final_response(),
        ]
    )
    client = FakeClient(adapter)

    await client.send_prompt_autoTC(MemoryManager(), "call", provider="openai", model="test-model", tools=[tool])

    error_payload = json.loads(adapter.payloads[1]["messages"][-1]["content"])
    assert error_payload["type"] == "ToolExecutionError"
    assert error_payload["tool"] == "known"
    assert expected_error in error_payload["error"]


@pytest.mark.asyncio
async def test_send_prompt_autoTC_returns_structured_unknown_tool_error() -> None:
    adapter = FakeAdapter(
        [
            _tool_call_response(call_id="call_missing", name="missing", arguments={}),
            _final_response(),
        ]
    )
    client = FakeClient(adapter)

    await client.send_prompt_autoTC(MemoryManager(), "call", provider="openai", model="test-model", tools=[])

    error_payload = json.loads(adapter.payloads[1]["messages"][-1]["content"])
    assert error_payload == {
        "error": "Unknown tool: missing",
        "tool": "missing",
        "type": "ToolExecutionError",
    }


@pytest.mark.asyncio
async def test_send_prompt_autoTC_max_tool_rounds_raises() -> None:
    adapter = FakeAdapter(
        [
            _tool_call_response(call_id="call_1", name="echo", arguments={"value": "a"}),
            _tool_call_response(call_id="call_2", name="echo", arguments={"value": "b"}),
        ]
    )
    client = FakeClient(adapter)
    tool = Tool(
        name="echo",
        description="Echo value.",
        parameters={"type": "object", "properties": {"value": {"type": "string"}}},
        func=lambda value: value,
    )

    with pytest.raises(RuntimeError, match="Maximum tool-call rounds exceeded: 1"):
        await client.send_prompt_autoTC(
            MemoryManager(),
            "echo",
            provider="openai",
            model="test-model",
            tools=[tool],
            max_tool_rounds=1,
        )


@pytest.mark.asyncio
async def test_send_prompt_autoTC_terminal_tool_returns_without_followup_request() -> None:
    adapter = FakeAdapter(
        [
            _tool_call_response(call_id="call_done", name="confirm", arguments={}),
            _tool_call_response(call_id="call_extra", name="confirm", arguments={}),
        ]
    )
    client = FakeClient(adapter)
    tool = Tool(
        name="confirm",
        description="Confirm final decision.",
        parameters={"type": "object", "properties": {}},
        func=lambda: {"committed": True},
    )

    response = await client.send_prompt_autoTC(
        MemoryManager(),
        "confirm",
        provider="openai",
        model="test-model",
        tools=[tool],
        tool_choice="required",
        terminal_tools={"confirm"},
    )

    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "confirm"
    assert len(adapter.payloads) == 1


@pytest.mark.asyncio
async def test_send_prompt_autoTC_commits_full_exchange_only_for_stateful_user_prompt() -> None:
    adapter = FakeAdapter(
        [
            _tool_call_response(call_id="call_123", name="echo", arguments={"value": "a"}),
            _final_response("final"),
        ]
    )
    client = FakeClient(adapter)
    memory = MemoryManager(stateful=True)
    tool = Tool(
        name="echo",
        description="Echo value.",
        parameters={"type": "object", "properties": {"value": {"type": "string"}}},
        func=lambda value: value,
    )

    await client.send_prompt_autoTC(memory, "echo", provider="openai", model="test-model", tools=[tool])

    history = memory.get_conversation_history()
    assert [message["role"] for message in history] == ["user", "assistant", "tool", "assistant"]
    assert history[2]["tool_call_id"] == "call_123"


@pytest.mark.asyncio
async def test_send_prompt_autoTC_does_not_mutate_memory_for_explicit_conversation_messages() -> None:
    adapter = FakeAdapter([_final_response("final")])
    client = FakeClient(adapter)
    memory = MemoryManager(stateful=True)

    await client.send_prompt_autoTC(
        memory,
        conversation_messages=[{"role": "user", "content": "hello"}],
        provider="openai",
        model="test-model",
        tools=[],
    )

    assert memory.get_conversation_history() == []


@pytest.mark.asyncio
async def test_send_prompt_existing_method_remains_single_request() -> None:
    adapter = FakeAdapter([_final_response("plain")])
    client = FakeClient(adapter)

    response = await client.send_prompt(MemoryManager(), "hello", provider="openai", model="test-model")

    assert response.text == "plain"
    assert len(adapter.payloads) == 1


@pytest.mark.asyncio
async def test_pooled_llm_client_forwards_send_prompt_autoTC() -> None:
    adapter = FakeAdapter(
        [
            _tool_call_response(call_id="call_pool", name="echo", arguments={"value": "pooled"}),
            _final_response("pooled final"),
        ]
    )
    raw_client = FakeClient(adapter)
    pooled = PooledLLMClient(raw_client, pool=asyncio.Semaphore(1))
    tool = Tool(
        name="echo",
        description="Echo value.",
        parameters={"type": "object", "properties": {"value": {"type": "string"}}},
        func=lambda value: value,
    )

    response = await pooled.send_prompt_autoTC(
        MemoryManager(),
        "echo",
        provider="openai",
        model="test-model",
        tools=[tool],
    )

    assert response.text == "pooled final"
    assert adapter.payloads[1]["messages"][-1]["tool_call_id"] == "call_pool"
