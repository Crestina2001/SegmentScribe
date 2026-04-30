from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ProviderName = Literal["openai", "minimax", "anthropic", "gemini", "deepseek"]


@dataclass(slots=True, frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(slots=True, frozen=True)
class ToolCall:
    id: str | None
    name: str
    arguments: dict[str, Any]
    raw_arguments: str | None = None


@dataclass(slots=True)
class PromptRequest:
    provider: ProviderName
    model: str
    messages: list[dict[str, Any]]
    reasoning_effort: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    response_format: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    trace_name: str | None = None
    session_id: str | None = None
    user_id: str | None = None


@dataclass(slots=True)
class PromptResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    provider: ProviderName | None = None
    model: str | None = None
    usage: dict[str, Any] | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    reasoning: str | None = None
    response_id: str | None = None
