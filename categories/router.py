from __future__ import annotations

from utils.heuristics import CATEGORY_ORDER, classify_heuristically, is_retryable_model_error, rank_models
from utils.models import RouteDecision, RuntimeConfig, TaskCategory, UsageTotals
from utils.parsing import parse_json_loose


class TaskRouter:
    """Route tasks with a cheap heuristic first, then a minimal model fallback."""

    def __init__(self, client, config: RuntimeConfig, usage: UsageTotals | None = None):
        self.client = client
        self.config = config
        self.usage = usage

    def _record_usage(self, usage: tuple[int, int]) -> None:
        if self.usage is not None:
            self.usage.add(*usage)

    def route(self, prompt: str) -> RouteDecision:
        heuristic = classify_heuristically(prompt)
        if heuristic.category != TaskCategory.UNKNOWN.value and heuristic.confidence >= 0.70:
            return heuristic
        return self._llm_route(prompt, heuristic)

    def _llm_route(self, prompt: str, heuristic: RouteDecision | None = None) -> RouteDecision:
        categories = ", ".join(category.value for category in CATEGORY_ORDER)
        system = (
            "You are a strict router for a hackathon AI agent. "
            "Classify the task into exactly one category and return JSON only."
        )
        user_parts = [
            f"Available categories: {categories}.",
            "Return JSON with keys category, confidence, reason.",
            "Category must be one of the available categories exactly.",
        ]
        if heuristic is not None:
            user_parts.append(
                f"Heuristic guess: {heuristic.category} (confidence {heuristic.confidence:.2f})."
            )
        user_parts.append("Task prompt:")
        user_parts.append(prompt)

        last_error: Exception | None = None
        for model in rank_models(self.config.allowed_models, "router"):
            try:
                text, usage, _ = self.client.chat_completion(
                    model,
                    [
                        {"role": "system", "content": system},
                        {"role": "user", "content": "\n\n".join(user_parts)},
                    ],
                    temperature=0.0,
                    max_tokens=96,
                )
                self._record_usage(usage)
                break
            except Exception as exc:
                last_error = exc
                if not is_retryable_model_error(exc):
                    raise
        else:
            raise RuntimeError(f"All router models failed: {last_error}")

        try:
            data = parse_json_loose(text)
            category = str(data.get("category", TaskCategory.UNKNOWN.value)).strip()
            confidence = float(data.get("confidence", 0.5))
            allowed_categories = {item.value for item in CATEGORY_ORDER}
            if category not in allowed_categories:
                category = TaskCategory.UNKNOWN.value
            return RouteDecision(category=category, confidence=confidence, source="llm-router")
        except Exception:
            return heuristic or RouteDecision(TaskCategory.UNKNOWN.value, 0.0, "router-fallback")
