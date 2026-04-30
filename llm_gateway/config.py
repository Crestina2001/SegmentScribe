from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from .model_resolution import (
    OFFICIAL_MODEL_PROVIDER_MAP,
    OFFICIAL_MODEL_PROVIDER_RULES,
    normalize_model_name,
)
from .models import ProviderName


@dataclass(slots=True)
class ProviderConfig:
    api_key: str | None = None
    base_url: str | None = None
    default_model: str | None = None
    models: tuple[str, ...] = ()
    request_timeout: float = 60.0
    http_proxy: str | None = None
    https_proxy: str | None = None

    @property
    def network_proxy(self) -> str | None:
        return self.https_proxy or self.http_proxy


@dataclass(slots=True)
class RetryConfig:
    max_retries: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 8.0
    jitter_seconds: float = 0.25


@dataclass(slots=True)
class ConcurrencyConfig:
    global_limit: int = 20
    provider_limits: dict[ProviderName, int] = field(default_factory=dict)
    model_limits: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class LangfuseConfig:
    public_key: str | None = None
    secret_key: str | None = None
    base_url: str | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.public_key and self.secret_key and self.base_url)


@dataclass(slots=True)
class GatewayConfig:
    openai: ProviderConfig
    minimax: ProviderConfig
    anthropic: ProviderConfig = field(default_factory=ProviderConfig)
    gemini: ProviderConfig = field(default_factory=ProviderConfig)
    deepseek: ProviderConfig = field(default_factory=ProviderConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)
    langfuse: LangfuseConfig = field(default_factory=LangfuseConfig)
    env_path: Path | None = None

    def resolve_provider(self, model: str, provider: ProviderName | None = None) -> ProviderName:
        if provider is not None:
            return provider

        normalized_model = normalize_model_name(model)
        if not normalized_model:
            raise ValueError("Model name must not be empty.")

        configured_map = self._configured_model_provider_map()
        configured_provider = configured_map.get(normalized_model)
        if configured_provider is not None:
            return configured_provider

        official_provider = OFFICIAL_MODEL_PROVIDER_MAP.get(normalized_model)
        if official_provider is not None:
            return official_provider

        rule_matches = {
            provider_name
            for resolver in OFFICIAL_MODEL_PROVIDER_RULES
            if (provider_name := resolver(normalized_model)) is not None
        }
        matches = sorted(rule_matches)
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise ValueError(
                f"Provider of Model '{model}' cannot be auto-resolved, please pass the Provider explicitly."
            )
        raise ValueError(
            f"Model '{model}' matches multiple provider resolution rules ({', '.join(matches)}). "
            "Pass provider explicitly."
        )

    def _configured_model_provider_map(self) -> dict[str, ProviderName]:
        configured: dict[str, ProviderName] = {}
        ambiguous: set[str] = set()
        for provider_name, provider_config in (
            ("openai", self.openai),
            ("minimax", self.minimax),
            ("anthropic", self.anthropic),
            ("gemini", self.gemini),
            ("deepseek", self.deepseek),
        ):
            for model_name in provider_config.models:
                normalized = normalize_model_name(model_name)
                if not normalized:
                    continue
                existing = configured.get(normalized)
                if existing is None:
                    configured[normalized] = provider_name
                    continue
                if existing != provider_name:
                    ambiguous.add(normalized)
        if ambiguous:
            model_list = ", ".join(sorted(ambiguous))
            raise ValueError(
                f"Configured model mappings are ambiguous for: {model_list}. Pass provider explicitly."
            )
        return configured


def load_gateway_config(env_path: str | os.PathLike[str] | None = None) -> GatewayConfig:
    resolved_env = Path(env_path) if env_path else Path.cwd() / ".env"
    if resolved_env.exists():
        load_dotenv(resolved_env, override=False)

    openai_base = _normalize_base_url(
        _env_first("OPENAI_BASE_URL", "OPENAI_PROXY_URL"),
        default="https://api.openai.com/v1",
    )
    minimax_base = _normalize_base_url(
        _env_first("MINIMAX_BASE_URL", "MINIMAX_API_HOST"),
        default="https://api.minimaxi.com/v1",
    )
    anthropic_base = _normalize_base_url(
        _env_first("ANTHROPIC_PROXY_URL", "ANTHROPIC_BASE_URL"),
        default="https://api.anthropic.com/v1",
    )
    gemini_base = _normalize_base_url(
        _env_first("GEMINI_PROXY_URL", "GEMINI_BASE_URL"),
        default="https://generativelanguage.googleapis.com/v1beta/openai",
    )
    deepseek_base = _normalize_base_url(
        _env_first("DEEPSEEK_PROXY_URL", "DEEPSEEK_BASE_URL"),
        default="https://api.deepseek.com/v1",
    )

    return GatewayConfig(
        openai=ProviderConfig(
            api_key=_env_first("OPENAI_API_KEY", "OPENAI_KEY"),
            base_url=openai_base,
            default_model=_env_first("OPENAI_MODEL"),
            models=_parse_model_list(_env_first("OPENAI_MODELS")),
            request_timeout=_env_float("OPENAI_TIMEOUT", default=60.0),
            http_proxy=_env_first("OPENAI_HTTP_PROXY", "OPENAI_PROXY"),
            https_proxy=_env_first("OPENAI_HTTPS_PROXY"),
        ),
        minimax=ProviderConfig(
            api_key=_env_first("MINIMAX_API_KEY", "MINIMAX_KEY"),
            base_url=minimax_base,
            default_model=_env_first("MINIMAX_MODEL"),
            models=_parse_model_list(_env_first("MINIMAX_MODELS")),
            request_timeout=_env_float("MINIMAX_TIMEOUT", default=60.0),
            http_proxy=_env_first("MINIMAX_HTTP_PROXY", "MINIMAX_PROXY"),
            https_proxy=_env_first("MINIMAX_HTTPS_PROXY"),
        ),
        anthropic=ProviderConfig(
            api_key=_env_first("ANTHROPIC_API_KEY", "ANTHROPIC_KEY"),
            base_url=anthropic_base,
            default_model=_env_first("ANTHROPIC_MODEL"),
            models=_parse_model_list(_env_first("ANTHROPIC_MODELS")),
            request_timeout=_env_float("ANTHROPIC_TIMEOUT", default=60.0),
            http_proxy=_env_first("ANTHROPIC_HTTP_PROXY", "ANTHROPIC_PROXY"),
            https_proxy=_env_first("ANTHROPIC_HTTPS_PROXY"),
        ),
        gemini=ProviderConfig(
            api_key=_env_first("GEMINI_API_KEY", "GEMINI_KEY"),
            base_url=gemini_base,
            default_model=_env_first("GEMINI_MODEL"),
            models=_parse_model_list(_env_first("GEMINI_MODELS")),
            request_timeout=_env_float("GEMINI_TIMEOUT", default=60.0),
            http_proxy=_env_first("GEMINI_HTTP_PROXY", "GEMINI_PROXY"),
            https_proxy=_env_first("GEMINI_HTTPS_PROXY"),
        ),
        deepseek=ProviderConfig(
            api_key=_env_first("DEEPSEEK_API_KEY", "DEEPSEEK_KEY"),
            base_url=deepseek_base,
            default_model=_env_first("DEEPSEEK_MODEL"),
            models=_parse_model_list(_env_first("DEEPSEEK_MODELS")),
            request_timeout=_env_float("DEEPSEEK_TIMEOUT", default=60.0),
            http_proxy=_env_first("DEEPSEEK_HTTP_PROXY", "DEEPSEEK_PROXY"),
            https_proxy=_env_first("DEEPSEEK_HTTPS_PROXY"),
        ),
        retry=RetryConfig(
            max_retries=_env_int("LLM_GATEWAY_MAX_RETRIES", default=3),
            base_delay_seconds=_env_float("LLM_GATEWAY_RETRY_BASE_DELAY", default=1.0),
            max_delay_seconds=_env_float("LLM_GATEWAY_RETRY_MAX_DELAY", default=8.0),
            jitter_seconds=_env_float("LLM_GATEWAY_RETRY_JITTER", default=0.25),
        ),
        concurrency=ConcurrencyConfig(
            global_limit=_env_int("LLM_GATEWAY_MAX_CONCURRENCY", default=20),
            provider_limits={
                "openai": _env_int("OPENAI_MAX_CONCURRENCY", default=10),
                "minimax": _env_int("MINIMAX_MAX_CONCURRENCY", default=10),
                "anthropic": _env_int("ANTHROPIC_MAX_CONCURRENCY", default=10),
                "gemini": _env_int("GEMINI_MAX_CONCURRENCY", default=10),
                "deepseek": _env_int("DEEPSEEK_MAX_CONCURRENCY", default=10),
            },
            model_limits=_parse_model_limits(_env_first("LLM_GATEWAY_MODEL_CONCURRENCY")),
        ),
        langfuse=LangfuseConfig(
            public_key=_env_first("LANGFUSE_PUBLIC_KEY"),
            secret_key=_env_first("LANGFUSE_SECRET_KEY"),
            base_url=_env_first("LANGFUSE_BASE_URL", "LANGFUSE_HOST"),
        ),
        env_path=resolved_env if resolved_env.exists() else None,
    )


def _parse_model_limits(raw_value: str | None) -> dict[str, int]:
    if not raw_value:
        return {}
    limits: dict[str, int] = {}
    for chunk in raw_value.split(","):
        item = chunk.strip()
        if not item or ":" not in item:
            continue
        key, value = item.rsplit(":", 1)
        try:
            limits[key.strip()] = int(value.strip())
        except ValueError:
            continue
    return limits


def _parse_model_list(raw_value: str | None) -> tuple[str, ...]:
    if not raw_value:
        return ()
    seen: set[str] = set()
    parsed: list[str] = []
    for chunk in raw_value.split(","):
        model = chunk.strip()
        normalized = normalize_model_name(model)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        parsed.append(model)
    return tuple(parsed)


def _normalize_base_url(raw_value: str | None, *, default: str) -> str:
    value = (raw_value or "").strip()
    if not value:
        return default
    normalized = value.rstrip("/")
    if normalized.endswith("/chat/completions"):
        normalized = normalized[: -len("/chat/completions")]
    if normalized.endswith("/messages"):
        normalized = normalized[: -len("/messages")]
    if normalized.endswith("/v1") or "/v1/" in normalized:
        return normalized
    if normalized.endswith("/openai") or "/v1beta/" in normalized:
        return normalized
    return f"{normalized}/v1"


def _env_first(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return value.strip()
    return None


def _env_int(name: str, *, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value.strip())
    except ValueError:
        return default


def _env_float(name: str, *, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value.strip())
    except ValueError:
        return default
