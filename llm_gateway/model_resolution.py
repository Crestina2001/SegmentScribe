from __future__ import annotations

from collections.abc import Callable

from .models import ProviderName

ModelProviderResolver = Callable[[str], ProviderName | None]


OFFICIAL_MODEL_PROVIDER_MAP: dict[str, ProviderName] = {
    # OpenAI official names seen in current model docs and latest-model guidance.
    "gpt-5": "openai",
    "gpt-5-mini": "openai",
    "gpt-5-nano": "openai",
    "gpt-5-pro": "openai",
    "gpt-5-codex": "openai",
    "gpt-5.1": "openai",
    "gpt-5.1-codex": "openai",
    "gpt-5.1-codex-mini": "openai",
    "gpt-5.1-codex-max": "openai",
    "gpt-5.2": "openai",
    "gpt-5.2-codex": "openai",
    "gpt-5.3-codex": "openai",
    "gpt-5.4": "openai",
    "gpt-5.4-mini": "openai",
    "gpt-5.4-nano": "openai",
    "gpt-4.1": "openai",
    "gpt-4.1-mini": "openai",
    "gpt-4.1-nano": "openai",
    "gpt-4o": "openai",
    "gpt-4o-mini": "openai",
    "gpt-4": "openai",
    "gpt-3.5-turbo": "openai",
    "gpt-realtime": "openai",
    "gpt-realtime-mini": "openai",
    "gpt-audio": "openai",
    "gpt-audio-mini": "openai",
    "gpt-4o-mini-transcribe": "openai",
    "gpt-image-1": "openai",
    "codex-mini-latest": "openai",
    "o1": "openai",
    "o1-mini": "openai",
    "o1-preview": "openai",
    "o3": "openai",
    "o3-mini": "openai",
    "o4-mini": "openai",
    "text-embedding-3-small": "openai",
    "text-embedding-3-large": "openai",
    "text-embedding-ada-002": "openai",
    "omni-moderation-latest": "openai",
    "omni-moderation-2024-09-26": "openai",
    # MiniMax official text-generation names from current docs.
    "minimax-m2.7": "minimax",
    "minimax-m2.7-highspeed": "minimax",
    "minimax-m2.5": "minimax",
    "minimax-m2.5-highspeed": "minimax",
    "minimax-m2.1": "minimax",
    "minimax-m2.1-highspeed": "minimax",
    "minimax-m2": "minimax",
    # Anthropic Claude models from current official model and release docs.
    "claude-opus-4-6": "anthropic",
    "claude-sonnet-4-6": "anthropic",
    "claude-sonnet-4-5-20250929": "anthropic",
    "claude-haiku-4-5-20251001": "anthropic",
    # Gemini official models from current Google AI docs.
    "gemini-3.1-pro-preview": "gemini",
    "gemini-3.1-pro-preview-customtools": "gemini",
    "gemini-3-flash-preview": "gemini",
    "gemini-3.1-flash-lite-preview": "gemini",
    "gemini-3.1-flash-live-preview": "gemini",
    "gemini-2.5-pro": "gemini",
    "gemini-2.5-flash": "gemini",
    "gemini-2.5-flash-lite": "gemini",
    "gemini-2.5-flash-native-audio-preview-12-2025": "gemini",
    "gemini-2.5-flash-preview-tts": "gemini",
    "gemini-2.5-pro-preview-tts": "gemini",
    # DeepSeek official chat-completions model aliases from current API docs.
    "deepseek-chat": "deepseek",
    "deepseek-reasoner": "deepseek",
    "deepseek-coder": "deepseek",
}


def resolve_openai_model_family(model: str) -> ProviderName | None:
    if "gpt" in model or model.startswith(("codex-", "text-embedding-", "omni-moderation-")):
        return "openai"
    return None


def resolve_minimax_model_family(model: str) -> ProviderName | None:
    if "minimax" in model:
        return "minimax"
    return None


def resolve_anthropic_model_family(model: str) -> ProviderName | None:
    if "claude" in model:
        return "anthropic"
    return None


def resolve_gemini_model_family(model: str) -> ProviderName | None:
    if "gemini" in model:
        return "gemini"
    return None


def resolve_deepseek_model_family(model: str) -> ProviderName | None:
    if "deepseek" in model:
        return "deepseek"
    return None


OFFICIAL_MODEL_PROVIDER_RULES: tuple[ModelProviderResolver, ...] = (
    resolve_openai_model_family,
    resolve_minimax_model_family,
    resolve_anthropic_model_family,
    resolve_gemini_model_family,
    resolve_deepseek_model_family,
)


def normalize_model_name(model: str) -> str:
    return model.strip().lower()
