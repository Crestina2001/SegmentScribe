from __future__ import annotations

from contextlib import ExitStack, contextmanager, nullcontext
from typing import Any

from .config import LangfuseConfig

try:
    from langfuse import get_client, propagate_attributes
except Exception:  # pragma: no cover - optional dependency at runtime
    get_client = None  # type: ignore[assignment]
    propagate_attributes = None  # type: ignore[assignment]


class LangfuseTracer:
    def __init__(self, config: LangfuseConfig) -> None:
        self._enabled = bool(config.enabled and get_client is not None)
        self._client = get_client() if self._enabled else None

    def start_generation(
        self,
        *,
        name: str,
        model: str,
        input_payload: Any,
        model_parameters: dict[str, Any],
        metadata: dict[str, Any] | None,
        user_id: str | None,
        session_id: str | None,
    ):
        if not self._enabled or self._client is None:
            return nullcontext(None)
        return self._generation_context(
            name=name,
            model=model,
            input_payload=input_payload,
            model_parameters=model_parameters,
            metadata=metadata,
            user_id=user_id,
            session_id=session_id,
        )

    @staticmethod
    def update_success(
        observation: Any,
        *,
        output: Any,
        usage: dict[str, Any] | None,
        metadata: dict[str, Any] | None,
    ) -> None:
        if observation is None:
            return
        observation.update(output=output, usage_details=usage, metadata=metadata)

    @staticmethod
    def update_error(observation: Any, *, error: Exception, metadata: dict[str, Any] | None) -> None:
        if observation is None:
            return
        observation.update(level="ERROR", status_message=str(error), metadata=metadata)

    def flush(self) -> None:
        if self._client is None:
            return
        try:
            self._client.flush()
        except Exception:
            pass

    @contextmanager
    def _generation_context(
        self,
        *,
        name: str,
        model: str,
        input_payload: Any,
        model_parameters: dict[str, Any],
        metadata: dict[str, Any] | None,
        user_id: str | None,
        session_id: str | None,
    ):
        stack = ExitStack()
        observation = None
        try:
            attributes = {
                "user_id": user_id,
                "session_id": session_id,
                "metadata": metadata,
            }
            cleaned = {key: value for key, value in attributes.items() if value is not None}
            if cleaned and propagate_attributes is not None:
                stack.enter_context(propagate_attributes(**cleaned))
            observation = stack.enter_context(
                self._client.start_as_current_observation(
                    as_type="generation",
                    name=name,
                    model=model,
                    input=input_payload,
                    model_parameters={key: value for key, value in model_parameters.items() if value is not None},
                    metadata=metadata,
                )
            )
        except Exception:
            observation = None
        try:
            yield observation
        finally:
            stack.close()
