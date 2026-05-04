from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any, Iterable

from .models import Tool, ToolCall, ToolDefinition

STRUCTURED_OUTPUT_TOOL_NAME = "submit_structured_output"
STRUCTURED_OUTPUT_TOOL_DESCRIPTION = "Submit the final response as a JSON object matching the requested schema."


class ToolCallManager:
    """Provider-agnostic tool definition and tool call normalization."""

    def build_structured_output_tool(
        self,
        schema: dict[str, Any],
        *,
        name: str = STRUCTURED_OUTPUT_TOOL_NAME,
        description: str = STRUCTURED_OUTPUT_TOOL_DESCRIPTION,
    ) -> dict[str, Any]:
        if not isinstance(schema, dict) or not schema:
            raise ValueError("output_schema must be a non-empty JSON-schema mapping.")
        normalized_name = str(name).strip()
        if not normalized_name:
            raise ValueError("Structured output tool name cannot be empty.")
        return {
            "type": "function",
            "function": {
                "name": normalized_name,
                "description": str(description).strip(),
                "parameters": dict(schema),
            },
        }

    def normalize_definitions(
        self,
        tools: Iterable[Tool | ToolDefinition | dict[str, Any]] | None,
    ) -> list[dict[str, Any]] | None:
        if not tools:
            return None

        normalized: list[dict[str, Any]] = []
        for tool in tools:
            item = self._to_mapping(tool)
            if item.get("type") == "function" and isinstance(item.get("function"), dict):
                function = dict(item["function"])
            else:
                function = {
                    "name": item.get("name"),
                    "description": item.get("description", ""),
                    "parameters": item.get("parameters"),
                }

            name = str(function.get("name") or "").strip()
            description = str(function.get("description") or "").strip()
            parameters = function.get("parameters")
            if not name:
                raise ValueError("Tool definition requires a non-empty name.")
            if not isinstance(parameters, dict):
                raise ValueError(f"Tool '{name}' requires JSON-schema parameters.")

            normalized.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": description,
                        "parameters": parameters,
                    },
                }
            )
        return normalized

    def normalize_choice(self, tool_choice: str | dict[str, Any] | None) -> str | dict[str, Any] | None:
        if tool_choice is None:
            return None
        if isinstance(tool_choice, str):
            value = tool_choice.strip()
            if value not in {"auto", "none", "required"}:
                raise ValueError(f"Unsupported tool_choice value: {tool_choice}")
            return value
        if not isinstance(tool_choice, dict):
            raise ValueError("tool_choice must be a string or mapping.")

        if tool_choice.get("type") == "function" and isinstance(tool_choice.get("function"), dict):
            function = dict(tool_choice["function"])
        elif isinstance(tool_choice.get("name"), str):
            function = {"name": tool_choice["name"]}
        else:
            raise ValueError("tool_choice mapping must specify a function name.")

        name = str(function.get("name") or "").strip()
        if not name:
            raise ValueError("tool_choice function name cannot be empty.")
        return {"type": "function", "function": {"name": name}}

    def parse_tool_calls(self, message: dict[str, Any] | None) -> list[ToolCall]:
        if not isinstance(message, dict):
            return []

        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            normalized_calls: list[ToolCall] = []
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                function = call.get("function")
                if not isinstance(function, dict):
                    continue
                name = str(function.get("name") or "").strip()
                if not name:
                    continue
                raw_arguments = function.get("arguments")
                normalized_calls.append(
                    ToolCall(
                        id=str(call.get("id")) if call.get("id") is not None else None,
                        name=name,
                        arguments=self._parse_arguments(raw_arguments),
                        raw_arguments=str(raw_arguments) if raw_arguments is not None else None,
                    )
                )
            return normalized_calls

        function_call = message.get("function_call")
        if isinstance(function_call, dict):
            name = str(function_call.get("name") or "").strip()
            if name:
                raw_arguments = function_call.get("arguments")
                return [
                    ToolCall(
                        id=None,
                        name=name,
                        arguments=self._parse_arguments(raw_arguments),
                        raw_arguments=str(raw_arguments) if raw_arguments is not None else None,
                    )
                ]
        return []

    def assistant_message_for_memory(self, raw_message: dict[str, Any]) -> dict[str, Any]:
        message = dict(raw_message)
        if "content" not in message or message["content"] is None:
            message["content"] = ""
        return message

    @staticmethod
    def _parse_arguments(raw_arguments: Any) -> dict[str, Any]:
        if raw_arguments is None:
            return {}
        if isinstance(raw_arguments, dict):
            return raw_arguments
        if not isinstance(raw_arguments, str):
            return {}
        raw_arguments = raw_arguments.strip()
        if not raw_arguments:
            return {}
        try:
            parsed = json.loads(raw_arguments)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _to_mapping(value: Tool | ToolDefinition | dict[str, Any]) -> dict[str, Any]:
        if is_dataclass(value):
            return asdict(value)
        if isinstance(value, dict):
            return dict(value)
        raise ValueError(f"Unsupported tool definition type: {type(value)!r}")
