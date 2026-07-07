from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from .models import RuntimeConfig


class FireworksClient:
    """Minimal OpenAI-compatible client that talks directly to Fireworks."""

    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.base_url = config.base_url.rstrip("/")

    def _build_url(self, endpoint: str) -> str:
        endpoint = endpoint.lstrip("/")
        if self.base_url.endswith(endpoint):
            return self.base_url
        return f"{self.base_url}/{endpoint}"

    def _post_json(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = self._build_url(endpoint)
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(url, data=data, method="POST")
        request.add_header("Content-Type", "application/json")
        request.add_header("Accept", "application/json")
        request.add_header("Authorization", f"Bearer {self.config.api_key}")

        try:
            with urllib.request.urlopen(request, timeout=self.config.request_timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Fireworks request failed with HTTP {exc.code}: {body}") from exc

    @staticmethod
    def _extract_text(payload: dict[str, Any]) -> str:
        choices = payload.get("choices") or []
        if not choices:
            raise RuntimeError(f"Unexpected Fireworks response shape: {payload!r}")

        choice = choices[0]
        message = choice.get("message") or {}
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
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

        raise RuntimeError(f"Could not extract model text from response: {payload!r}")

    @staticmethod
    def _extract_usage(payload: dict[str, Any]) -> tuple[int, int]:
        usage = payload.get("usage") or {}
        return int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)

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
        payload: dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if top_p is not None:
            payload["top_p"] = top_p

        response = self._post_json(endpoint, payload)
        text = self._extract_text(response)
        usage = self._extract_usage(response)
        return text, usage, response
