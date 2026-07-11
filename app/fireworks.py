from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


DEFAULT_BASE_URL = "https://api.fireworks.ai/inference/v1"


class FireworksError(RuntimeError):
    pass


@dataclass
class FireworksResponse:
    content: str
    usage: dict[str, int] | None = None
    raw: dict[str, Any] | None = None
    finish_reason: str | None = None


class FireworksClient:
    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        self.api_key = api_key or os.environ.get("FIREWORKS_API_KEY", "")
        self.base_url = (base_url or os.environ.get("FIREWORKS_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        if not self.api_key:
            raise FireworksError("FIREWORKS_API_KEY is missing")
        self._chat_url = f"{self.base_url}/chat/completions"

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 512,
        response_format: dict[str, Any] | None = None,
        timeout: int = 30,
        retries: int = 2,
    ) -> FireworksResponse:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                req = urllib.request.Request(self._chat_url, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                content = data["choices"][0]["message"].get("content", "")
                usage = data.get("usage")
                finish_reason = None
                try:
                    finish_reason = data["choices"][0].get("finish_reason")
                except Exception:
                    finish_reason = None
                return FireworksResponse(content=content, usage=usage, raw=data, finish_reason=finish_reason)
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, KeyError) as exc:
                last_error = exc
                if attempt >= retries:
                    break
                time.sleep(0.5 * (attempt + 1))
        raise FireworksError(f"Fireworks request failed: {last_error}")
