from __future__ import annotations

import inspect
import json
import base64
import mimetypes
from dataclasses import replace
from pathlib import Path
from typing import Any

from .concurrency import ConcurrencyController
from .config import GatewayConfig, load_gateway_config
from .memory_manager import MemoryManager
from .models import PromptRequest, PromptResponse, ProviderName, Tool, ToolCall, ToolDefinition
from .providers import AnthropicAdapter, BaseProviderAdapter, DeepSeekAdapter, GeminiAdapter, MiniMaxAdapter, OpenAIAdapter
from .retry import execute_with_retry
from .telemetry import LangfuseTracer
from .tools import STRUCTURED_OUTPUT_TOOL_NAME, ToolCallManager


class UnifiedClient:
    """Single async client for sending prompts to supported providers."""

    def __init__(self, *, env_path: str | None = None, config: GatewayConfig | None = None) -> None:
        self.config = config or load_gateway_config(env_path)
        self.tool_manager = ToolCallManager()
        self.concurrency = ConcurrencyController(self.config.concurrency)
        self.tracer = LangfuseTracer(self.config.langfuse)
        self._adapters: dict[ProviderName, BaseProviderAdapter] = {}

    async def send_prompt(
        self,
        memory_manager: MemoryManager,
        user_prompt: str | None = None,
        *,
        conversation_messages: list[dict[str, Any]] | None = None,
        provider: ProviderName | None = None,
        model: str,
        reasoning_effort: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        response_format: dict[str, Any] | None = None,
        images: list[str] | None = None,
        tools: list[Tool | ToolDefinition | dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        output_schema: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        trace_name: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> PromptResponse:
        provider_name = self.config.resolve_provider(model=model, provider=provider)
        request = self._build_request(
            memory_manager=memory_manager,
            user_prompt=user_prompt,
            conversation_messages=conversation_messages,
            provider_name=provider_name,
            model=model,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            response_format=response_format,
            images=images,
            tools=tools,
            tool_choice=tool_choice,
            output_schema=output_schema,
            metadata=metadata,
            trace_name=trace_name,
            session_id=session_id,
            user_id=user_id,
        )
        adapter = self._get_adapter(provider_name)
        payload = adapter.prepare_request(request)
        raw_response = await self._send_traced(
            adapter=adapter,
            request=request,
            payload=payload,
            metadata=metadata,
        )
        parsed = adapter.parse_response(raw_response)

        if conversation_messages is None:
            assistant_choice = (raw_response.get("choices") or [{}])[0]
            assistant_message = assistant_choice.get("message") if isinstance(assistant_choice, dict) else None
            if assistant_message is not None:
                memory_manager.commit_turn(
                    user_prompt=user_prompt,
                    assistant_message=self.tool_manager.assistant_message_for_memory(assistant_message),
                )
        return parsed

    async def send_prompt_raw(
        self,
        memory_manager: MemoryManager,
        user_prompt: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        provider_name = self.config.resolve_provider(model=kwargs["model"], provider=kwargs.get("provider"))
        request = self._build_request(
            memory_manager=memory_manager,
            user_prompt=user_prompt,
            conversation_messages=kwargs.get("conversation_messages"),
            provider_name=provider_name,
            model=kwargs["model"],
            reasoning_effort=kwargs.get("reasoning_effort"),
            temperature=kwargs.get("temperature"),
            max_tokens=kwargs.get("max_tokens"),
            top_p=kwargs.get("top_p"),
            response_format=kwargs.get("response_format"),
            images=kwargs.get("images"),
            tools=kwargs.get("tools"),
            tool_choice=kwargs.get("tool_choice"),
            output_schema=kwargs.get("output_schema"),
            metadata=kwargs.get("metadata"),
            trace_name=kwargs.get("trace_name"),
            session_id=kwargs.get("session_id"),
            user_id=kwargs.get("user_id"),
        )
        adapter = self._get_adapter(provider_name)
        payload = adapter.prepare_request(request)
        return await self._send_with_controls(
            adapter=adapter,
            provider=provider_name,
            model=request.model,
            payload=payload,
        )

    async def send_prompt_autoTC(
        self,
        memory_manager: MemoryManager,
        user_prompt: str | None = None,
        *,
        conversation_messages: list[dict[str, Any]] | None = None,
        provider: ProviderName | None = None,
        model: str,
        reasoning_effort: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        response_format: dict[str, Any] | None = None,
        images: list[str] | None = None,
        tools: list[Tool] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        output_schema: dict[str, Any] | None = None,
        max_tool_rounds: int = 5,
        metadata: dict[str, Any] | None = None,
        trace_name: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> PromptResponse:
        if output_schema is not None:
            raise ValueError("send_prompt_autoTC does not support output_schema because it uses tools internally.")
        if max_tool_rounds < 0:
            raise ValueError("max_tool_rounds must be greater than or equal to 0.")

        tools = tools or []
        tool_registry = self._build_tool_registry(tools)
        provider_name = self.config.resolve_provider(model=model, provider=provider)
        request = self._build_request(
            memory_manager=memory_manager,
            user_prompt=user_prompt,
            conversation_messages=conversation_messages,
            provider_name=provider_name,
            model=model,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            response_format=response_format,
            images=images,
            tools=tools,
            tool_choice=tool_choice,
            output_schema=None,
            metadata=metadata,
            trace_name=trace_name,
            session_id=session_id,
            user_id=user_id,
        )
        adapter = self._get_adapter(provider_name)
        messages = [dict(message) for message in request.messages]
        turn_messages: list[dict[str, Any]] = []
        prompt = (user_prompt or "").strip()
        if conversation_messages is None and prompt:
            turn_messages.append({"role": "user", "content": prompt})
        elif conversation_messages is None and images is not None:
            built_user = messages[-1] if messages and messages[-1].get("role") == "user" else None
            if built_user is not None:
                turn_messages.append(dict(built_user))

        for tool_round in range(max_tool_rounds + 1):
            round_request = replace(request, messages=messages)
            payload = adapter.prepare_request(round_request)
            raw_response = await self._send_traced(
                adapter=adapter,
                request=round_request,
                payload=payload,
                metadata=metadata,
            )
            parsed = adapter.parse_response(raw_response)
            assistant_message = self._assistant_message_from_raw(raw_response)

            if not parsed.tool_calls:
                if assistant_message is not None:
                    turn_messages.append(self.tool_manager.assistant_message_for_memory(assistant_message))
                if conversation_messages is None and memory_manager.stateful:
                    for message in turn_messages:
                        memory_manager.append_message(message)
                return parsed

            if tool_round >= max_tool_rounds:
                raise RuntimeError(f"Maximum tool-call rounds exceeded: {max_tool_rounds}")

            if assistant_message is None:
                assistant_message = self._assistant_message_from_response(parsed)
            assistant_for_memory = self.tool_manager.assistant_message_for_memory(assistant_message)
            messages.append(assistant_for_memory)
            turn_messages.append(dict(assistant_for_memory))

            for call in parsed.tool_calls:
                tool_message = await self._tool_result_message(call, tool_registry)
                messages.append(tool_message)
                turn_messages.append(dict(tool_message))

        raise RuntimeError(f"Maximum tool-call rounds exceeded: {max_tool_rounds}")

    async def close(self) -> None:
        for adapter in self._adapters.values():
            await adapter.close()
        self.tracer.flush()

    async def __aenter__(self) -> "UnifiedClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def _send_traced(
        self,
        *,
        adapter: BaseProviderAdapter,
        request: PromptRequest,
        payload: dict[str, Any],
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        generation_name = request.trace_name or f"{request.provider}.send_prompt"
        with self.tracer.start_generation(
            name=generation_name,
            model=request.model,
            input_payload=request.messages,
            model_parameters={
                "reasoning_effort": request.reasoning_effort,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "top_p": request.top_p,
            },
            metadata=metadata,
            user_id=request.user_id,
            session_id=request.session_id,
        ) as observation:
            try:
                raw_response = await self._send_with_controls(
                    adapter=adapter,
                    provider=request.provider,
                    model=request.model,
                    payload=payload,
                )
                parsed = adapter.parse_response(raw_response)
                self.tracer.update_success(
                    observation,
                    output=self._trace_output(raw_response=raw_response, parsed=parsed),
                    usage=parsed.usage,
                    metadata={"provider": parsed.provider, "model": parsed.model, **(metadata or {})},
                )
                return raw_response
            except Exception as exc:
                self.tracer.update_error(
                    observation,
                    error=exc,
                    metadata={"provider": request.provider, "model": request.model, **(metadata or {})},
                )
                raise

    async def _send_with_controls(
        self,
        *,
        adapter: BaseProviderAdapter,
        provider: ProviderName,
        model: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        async def operation() -> dict[str, Any]:
            async with self.concurrency.acquire(provider, model):
                return await adapter.send(payload)

        return await execute_with_retry(
            operation,
            config=self.config.retry,
            should_retry=adapter.is_retryable_error,
        )

    def _build_request(
        self,
        *,
        memory_manager: MemoryManager,
        user_prompt: str | None,
        conversation_messages: list[dict[str, Any]] | None,
        provider_name: ProviderName,
        model: str,
        reasoning_effort: str | None,
        temperature: float | None,
        max_tokens: int | None,
        top_p: float | None,
        response_format: dict[str, Any] | None,
        images: list[str] | None,
        tools: list[Tool | ToolDefinition | dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        output_schema: dict[str, Any] | None,
        metadata: dict[str, Any] | None,
        trace_name: str | None,
        session_id: str | None,
        user_id: str | None,
    ) -> PromptRequest:
        normalized_tools, normalized_tool_choice = self._resolve_tooling(
            tools=tools,
            tool_choice=tool_choice,
            output_schema=output_schema,
        )
        if conversation_messages is not None and images is not None:
            raise ValueError(
                "images cannot be combined with conversation_messages. Pass fully formed multimodal "
                "conversation_messages or use user_prompt with images."
            )
        if images is not None:
            conversation_messages = [
                {
                    "role": "user",
                    "content": self._build_image_content(user_prompt=user_prompt, images=images),
                }
            ]
            user_prompt = None

        return PromptRequest(
            provider=provider_name,
            model=model,
            messages=memory_manager.build_messages(
                user_prompt=user_prompt,
                conversation_messages=conversation_messages,
            ),
            reasoning_effort=reasoning_effort,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            response_format=response_format,
            tools=normalized_tools,
            tool_choice=normalized_tool_choice,
            metadata=metadata,
            trace_name=trace_name,
            session_id=session_id,
            user_id=user_id,
        )

    def _resolve_tooling(
        self,
        *,
        tools: list[Tool | ToolDefinition | dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        output_schema: dict[str, Any] | None,
    ) -> tuple[list[dict[str, Any]] | None, str | dict[str, Any] | None]:
        if output_schema is not None:
            if tools:
                raise ValueError(
                    "output_schema cannot be combined with tools. Choose either manual tools or the structured-output shortcut."
                )
            if tool_choice is not None:
                raise ValueError(
                    "output_schema cannot be combined with tool_choice. Choose either manual tools or the structured-output shortcut."
                )
            structured_tool = self.tool_manager.build_structured_output_tool(output_schema)
            return [structured_tool], {"type": "function", "function": {"name": STRUCTURED_OUTPUT_TOOL_NAME}}
        return (
            self.tool_manager.normalize_definitions(tools),
            self.tool_manager.normalize_choice(tool_choice),
        )

    @staticmethod
    def _build_tool_registry(tools: list[Tool]) -> dict[str, Tool]:
        registry: dict[str, Tool] = {}
        for tool in tools:
            if not isinstance(tool, Tool):
                raise ValueError("send_prompt_autoTC tools must be Tool instances.")
            name = tool.name.strip()
            if not name:
                raise ValueError("Tool name cannot be empty.")
            if name in registry:
                raise ValueError(f"Duplicate tool name: {name}")
            registry[name] = tool
        return registry

    async def _tool_result_message(self, call: ToolCall, tool_registry: dict[str, Tool]) -> dict[str, Any]:
        tool = tool_registry.get(call.name)
        if tool is None:
            content = self._tool_error_content(
                tool=call.name,
                error=f"Unknown tool: {call.name}",
                error_type="ToolExecutionError",
            )
        else:
            try:
                result = tool.func(**call.arguments)
                if inspect.isawaitable(result):
                    result = await result
                content = self._serialize_tool_result(tool, result)
            except Exception as exc:
                content = self._tool_error_content(
                    tool=call.name,
                    error=str(exc),
                    error_type="ToolExecutionError",
                )
        return {
            "role": "tool",
            "tool_call_id": call.id,
            "content": content,
        }

    @staticmethod
    def _serialize_tool_result(tool: Tool, result: Any) -> str:
        if tool.serialize is not None:
            serialized = tool.serialize(result)
            return serialized if isinstance(serialized, str) else str(serialized)
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False)

    @staticmethod
    def _tool_error_content(*, tool: str, error: str, error_type: str) -> str:
        return json.dumps(
            {
                "error": error,
                "tool": tool,
                "type": error_type,
            },
            ensure_ascii=False,
        )

    @staticmethod
    def _assistant_message_from_raw(raw_response: dict[str, Any]) -> dict[str, Any] | None:
        choices = raw_response.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            return None
        message = first_choice.get("message")
        return dict(message) if isinstance(message, dict) else None

    @staticmethod
    def _assistant_message_from_response(response: PromptResponse) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": response.text or "",
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": call.raw_arguments if call.raw_arguments is not None else json.dumps(call.arguments),
                    },
                }
                for call in response.tool_calls
            ],
        }

    def _build_image_content(self, *, user_prompt: str | None, images: list[str]) -> list[dict[str, Any]]:
        if not isinstance(images, list):
            raise ValueError("images must be a list of local paths or HTTP(S) URLs.")

        content: list[dict[str, Any]] = []
        prompt = (user_prompt or "").strip()
        if prompt:
            content.append({"type": "text", "text": prompt})

        for image in images:
            content.append({"type": "image_url", "image_url": {"url": self._image_to_url(image)}})
        return content

    @staticmethod
    def _image_to_url(image: str) -> str:
        if not isinstance(image, str) or not image.strip():
            raise ValueError("images must contain non-empty local paths or HTTP(S) URLs.")

        source = image.strip()
        lowered = source.lower()
        if lowered.startswith(("http://", "https://")):
            return source
        if "://" in source:
            raise ValueError(f"Unsupported image URI scheme for {source!r}; use a local path or HTTP(S) URL.")

        path = Path(source).expanduser()
        if not path.exists():
            raise ValueError(f"Image file does not exist: {source}")
        if not path.is_file():
            raise ValueError(f"Image path is not a file: {source}")

        mime_type, _ = mimetypes.guess_type(path.name)
        if not mime_type or not mime_type.startswith("image/"):
            raise ValueError(f"Image file must have a recognized image MIME type: {source}")

        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    @staticmethod
    def _trace_output(*, raw_response: dict[str, Any], parsed: Any) -> Any:
        """Return a Langfuse-friendly assistant output.

        Tool-only chat responses often have empty ``content`` but meaningful
        ``tool_calls``. Tracing only ``parsed.text`` hides that decision in
        Langfuse, so preserve the provider assistant message shape when
        possible.
        """
        choices = raw_response.get("choices")
        if isinstance(choices, list) and choices:
            messages: list[Any] = []
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                message = choice.get("message")
                if isinstance(message, dict):
                    messages.append(UnifiedClient._compact_assistant_message(message))
                elif "text" in choice:
                    messages.append(choice.get("text"))
            if len(messages) == 1:
                return messages[0]
            if messages:
                return messages

        output = raw_response.get("output")
        if output is not None:
            return output

        if getattr(parsed, "tool_calls", None):
            return {
                "role": "assistant",
                "content": getattr(parsed, "text", "") or "",
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": call.raw_arguments if call.raw_arguments is not None else call.arguments,
                        },
                    }
                    for call in parsed.tool_calls
                ],
            }
        return getattr(parsed, "text", "")

    @staticmethod
    def _compact_assistant_message(message: dict[str, Any]) -> dict[str, Any]:
        keep_keys = ("role", "content", "tool_calls", "function_call", "reasoning_content", "audio")
        compact = {key: message.get(key) for key in keep_keys if key in message and message.get(key) is not None}
        if "role" not in compact:
            compact["role"] = "assistant"
        if "content" not in compact:
            compact["content"] = ""
        return compact

    def _get_adapter(self, provider: ProviderName) -> BaseProviderAdapter:
        existing = self._adapters.get(provider)
        if existing is not None:
            return existing

        if provider == "openai":
            if not self.config.openai.api_key:
                raise ValueError("OPENAI_API_KEY or OPENAI_KEY is required for provider='openai'.")
            adapter: BaseProviderAdapter = OpenAIAdapter(self.config.openai, tool_manager=self.tool_manager)
        elif provider == "minimax":
            if not self.config.minimax.api_key:
                raise ValueError("MINIMAX_API_KEY or MINIMAX_KEY is required for provider='minimax'.")
            adapter = MiniMaxAdapter(self.config.minimax, tool_manager=self.tool_manager)
        elif provider == "anthropic":
            if not self.config.anthropic.api_key:
                raise ValueError("ANTHROPIC_API_KEY or ANTHROPIC_KEY is required for provider='anthropic'.")
            adapter = AnthropicAdapter(self.config.anthropic, tool_manager=self.tool_manager)
        elif provider == "gemini":
            if not self.config.gemini.api_key:
                raise ValueError("GEMINI_API_KEY or GEMINI_KEY is required for provider='gemini'.")
            adapter = GeminiAdapter(self.config.gemini, tool_manager=self.tool_manager)
        elif provider == "deepseek":
            if not self.config.deepseek.api_key:
                raise ValueError("DEEPSEEK_API_KEY or DEEPSEEK_KEY is required for provider='deepseek'.")
            adapter = DeepSeekAdapter(self.config.deepseek, tool_manager=self.tool_manager)
        else:  # pragma: no cover
            raise ValueError(f"Unsupported provider: {provider}")

        self._adapters[provider] = adapter
        return adapter
