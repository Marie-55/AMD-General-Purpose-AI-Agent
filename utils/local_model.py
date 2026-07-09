from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import RuntimeConfig


def _extract_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError(f"Unexpected local model response shape: {payload!r}")

    choice = choices[0]
    message = choice.get("message") or {}
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item, str):
                    parts.append(item)
            if parts:
                return "".join(parts)

    text = choice.get("text")
    if isinstance(text, str):
        return text

    raise RuntimeError(f"Could not extract local model text from response: {payload!r}")


class LocalModelClient:
    """Optional GGUF-backed local model adapter.

    The class is intentionally lazy: if a runtime or model file is unavailable,
    the agent can still operate with local rules and Fireworks fallback.
    """

    def __init__(
        self,
        *,
        model_path: Path | None,
        enabled: bool,
        n_ctx: int = 2048,
        n_threads: int = 2,
        default_max_tokens: int = 192,
    ):
        self.model_path = Path(model_path).expanduser() if model_path else None
        self.enabled = enabled and self.model_path is not None
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.default_max_tokens = default_max_tokens
        self._llama = None
        self._load_error: Exception | None = None

        if not self.enabled or self.model_path is None or not self.model_path.exists():
            self.enabled = False
            return

        try:
            from llama_cpp import Llama  # type: ignore

            self._llama = Llama(
                model_path=str(self.model_path),
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                verbose=False,
            )
        except Exception as exc:  # pragma: no cover - optional runtime path
            self._llama = None
            self._load_error = exc
            self.enabled = False

    @classmethod
    def from_config(cls, config: RuntimeConfig) -> "LocalModelClient":
        return cls(
            model_path=config.local_model_path,
            enabled=config.local_model_enabled,
            n_ctx=config.local_model_n_ctx,
            n_threads=config.local_model_n_threads,
            default_max_tokens=config.local_model_max_tokens,
        )

    @property
    def available(self) -> bool:
        return self.enabled and self._llama is not None

    @property
    def load_error(self) -> Exception | None:
        return self._load_error

    def chat_completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        top_p: float | None = None,
        endpoint: str = "chat/completions",
    ) -> tuple[str, tuple[int, int], dict[str, Any]]:
        del model, endpoint
        if not self.available or self._llama is None:
            raise RuntimeError("Local model runtime is unavailable.")

        response = self._llama.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens or self.default_max_tokens,
            **({"top_p": top_p} if top_p is not None else {}),
        )
        if isinstance(response, str):
            text = response
            payload = {"choices": [{"message": {"content": text}}]}
        else:
            text = _extract_text(response)
            payload = response

        # Local model tokens do not count toward the Fireworks score.
        return text, (0, 0), payload
