from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any

from .heuristics import classify_heuristically
from .models import RuntimeConfig
from .parsing import compact_text, normalize_text


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

            reasoning_content = message.get("reasoning_content")
            if isinstance(reasoning_content, str):
                return reasoning_content
            if isinstance(reasoning_content, list):
                parts = []
                for item in reasoning_content:
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


class MockFireworksClient:
    """Deterministic local fallback for development without Fireworks credentials."""

    def __init__(self, config: RuntimeConfig):
        self.config = config

    @staticmethod
    def _usage(text: str) -> tuple[int, int]:
        prompt_tokens = max(1, len(text) // 16)
        completion_tokens = max(1, len(text) // 24)
        return prompt_tokens, completion_tokens

    def _respond_router(self, prompt: str) -> str:
        decision = classify_heuristically(prompt)
        return json.dumps(
            {
                "category": decision.category,
                "confidence": round(decision.confidence, 2),
                "reason": f"mock-router:{decision.source}",
            },
            ensure_ascii=False,
        )

    def _respond_sentiment(self, prompt: str) -> str:
        text = prompt.lower()
        if any(word in text for word in ("love", "great", "excellent", "amazing", "wonderful", "good")):
            label = "positive"
        elif any(word in text for word in ("hate", "bad", "terrible", "awful", "poor", "worst")):
            label = "negative"
        else:
            label = "neutral"
        return json.dumps({"label": label, "reason": "mock sentiment classification"}, ensure_ascii=False)

    def _respond_ner(self, prompt: str) -> str:
        candidates = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", prompt)
        entities = [{"text": item, "type": "other"} for item in dict.fromkeys(candidates)]
        return json.dumps({"entities": entities[:8]}, ensure_ascii=False)

    def _respond_summary(self, prompt: str) -> str:
        cleaned = normalize_text(prompt)
        return compact_text(cleaned[:240]) or "Mock summary unavailable."

    def _respond_code(self, prompt: str) -> str:
        if "debug" in prompt.lower() or "bug" in prompt.lower():
            return "```python\n# Mock code output for local development\npass\n```"
        return "```python\n# Mock code output for local development\nanswer = None\n```"

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
        del model, temperature, max_tokens, top_p, endpoint
        prompt = "\n".join(message.get("content", "") for message in messages)
        user_prompt = "\n".join(
            message.get("content", "") for message in messages if message.get("role") == "user"
        ) or prompt
        lower_prompt = prompt.lower()
        lower_user_prompt = user_prompt.lower()

        if "classify the task" in lower_prompt or "available categories" in lower_prompt:
            text = self._respond_router(prompt)
        elif "classify sentiment" in lower_prompt or '"label"' in lower_prompt:
            text = self._respond_sentiment(user_prompt)
        elif "extract named entities" in lower_prompt or '"entities"' in lower_prompt:
            text = self._respond_ner(user_prompt)
        elif "summarize" in lower_prompt or "summarise" in lower_prompt:
            text = self._respond_summary(user_prompt)
        elif "python 3 program" in lower_prompt or "return only executable python code" in lower_prompt:
            text = self._respond_code(user_prompt)
        else:
            text = f"Mock response: {compact_text(user_prompt)[:200]}"

        usage = self._usage(prompt + text)
        response = {"choices": [{"message": {"content": text}}], "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1]}}
        return text, usage, response
