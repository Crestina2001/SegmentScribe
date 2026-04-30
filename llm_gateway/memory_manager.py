from __future__ import annotations

from typing import Any, Iterable


class MemoryManager:
    """Owns conversational state independent of provider behavior."""

    def __init__(self, system_prompt: str | None = None, stateful: bool = False) -> None:
        self.system_prompt = (system_prompt or "").strip() or None
        self.stateful = stateful
        self._conversation_history: list[dict[str, Any]] = []

    def reset_conversation(self) -> None:
        self._conversation_history = []

    def append_message(self, message: dict[str, Any]) -> None:
        normalized = self._normalize_message(message)
        if normalized is not None:
            self._conversation_history.append(normalized)

    def extend_messages(self, messages: Iterable[dict[str, Any]]) -> None:
        for message in messages:
            self.append_message(message)

    def get_conversation_history(self) -> list[dict[str, Any]]:
        return [dict(message) for message in self._conversation_history]

    def build_messages(
        self,
        *,
        user_prompt: str | None = None,
        conversation_messages: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(self.get_conversation_history())

        if conversation_messages is not None:
            for message in conversation_messages:
                normalized = self._normalize_message(message)
                if normalized is not None:
                    messages.append(normalized)
            return messages

        prompt = (user_prompt or "").strip()
        if prompt:
            messages.append({"role": "user", "content": prompt})
        return messages

    def commit_turn(self, *, user_prompt: str | None, assistant_message: dict[str, Any] | None) -> None:
        if not self.stateful:
            return

        prompt = (user_prompt or "").strip()
        if prompt:
            self._conversation_history.append({"role": "user", "content": prompt})

        normalized = self._normalize_message(assistant_message)
        if normalized is not None:
            self._conversation_history.append(normalized)

    @staticmethod
    def _normalize_message(message: dict[str, Any] | None) -> dict[str, Any] | None:
        if not isinstance(message, dict):
            return None
        role = str(message.get("role") or "").strip().lower()
        if role not in {"system", "user", "assistant", "tool"}:
            return None

        has_assistant_payload = role == "assistant" and any(
            key in message for key in ("tool_calls", "function_call", "reasoning_details")
        )
        content = message.get("content")
        if isinstance(content, str):
            normalized_content: Any = content.strip()
            if not normalized_content and role != "tool" and not has_assistant_payload:
                return None
        elif isinstance(content, list):
            normalized_content = [dict(item) if isinstance(item, dict) else item for item in content]
        elif isinstance(content, dict):
            normalized_content = dict(content)
        elif content is None:
            normalized_content = ""
        else:
            normalized_content = str(content)

        normalized: dict[str, Any] = {"role": role, "content": normalized_content}
        if "tool_call_id" in message:
            normalized["tool_call_id"] = message["tool_call_id"]
        if "name" in message:
            normalized["name"] = message["name"]
        if "tool_calls" in message and isinstance(message["tool_calls"], list):
            normalized["tool_calls"] = [dict(item) if isinstance(item, dict) else item for item in message["tool_calls"]]
        if "function_call" in message and isinstance(message["function_call"], dict):
            normalized["function_call"] = dict(message["function_call"])
        if "reasoning_details" in message and isinstance(message["reasoning_details"], list):
            normalized["reasoning_details"] = [
                dict(item) if isinstance(item, dict) else item for item in message["reasoning_details"]
            ]
        return normalized
