"""
Regression test for the sentiment token-budget fix.

Caught in real runs: sentiment's Fireworks calls hardcoded max_tokens=60,
which was frequently exhausted by minimax-m3's internal reasoning pass
before any visible content was produced (fireworks_client.py's
"reasoning_exhausted_budget" case), or truncated the answer mid-sentence
right where it would have acknowledged the second side of a mixed review
(judge_guide.md T03/T03b). Both Fireworks call sites now read the budget
from config.CATEGORY_MAX_TOKENS instead of a hardcoded literal.
"""
from categories.handlers.context import PipelineContext
from categories.handlers.sentiment import handle
from categories.metrics import MetricsCollector
from categories.task_categories import SENTIMENT
from config import CATEGORY_MAX_TOKENS

TASK = {"task_id": "t1", "prompt": "Classify the sentiment of this review: mixed feelings overall."}


class _UnloadedLocalModel:
    def is_loaded(self) -> bool:
        return False


class _RecordingClient:
    def __init__(self, answer: str):
        self._answer = answer
        self.calls = []

    def call(self, model, messages, max_tokens=400, timeout=25, **kwargs):
        self.calls.append(max_tokens)
        lat = {
            "model": model, "ttft_ms": 1.0, "total_latency_ms": 2.0,
            "prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10, "error": None,
        }
        return self._answer, lat


def test_sentiment_fireworks_call_uses_configured_token_budget(monkeypatch):
    monkeypatch.setenv("ALLOWED_MODELS", "accounts/fireworks/models/minimax-m3")
    client = _RecordingClient("positive: good overall.")
    ctx = PipelineContext(client=client, metrics=MetricsCollector(), local_model=_UnloadedLocalModel())

    handle(TASK, SENTIMENT, ctx)

    assert client.calls  # at least one Fireworks call happened
    assert all(mt == CATEGORY_MAX_TOKENS[SENTIMENT] for mt in client.calls)
    assert CATEGORY_MAX_TOKENS[SENTIMENT] > 60  # the old, too-tight budget


def test_sentiment_forced_minimax_retry_also_uses_configured_budget(monkeypatch):
    monkeypatch.setenv("ALLOWED_MODELS", "accounts/fireworks/models/minimax-m3")
    # Empty/unlabeled first answer forces the "strict minimax" retry branch.
    client = _RecordingClient("")
    ctx = PipelineContext(client=client, metrics=MetricsCollector(), local_model=_UnloadedLocalModel())

    handle(TASK, SENTIMENT, ctx)

    assert len(client.calls) == 2  # primary attempt + forced strict retry
    assert all(mt == CATEGORY_MAX_TOKENS[SENTIMENT] for mt in client.calls)
