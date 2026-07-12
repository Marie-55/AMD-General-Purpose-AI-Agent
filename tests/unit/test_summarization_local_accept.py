"""
Regression tests for two summarization fixes from the same real run:

1. The local-accept gate required >=10 words, rejecting correctly-terse
   local summaries (e.g. compliant with "in one sentence" / "in exactly 5
   words" requests) and forcing unnecessary Fireworks escalation. Now only
   escalates on a genuinely empty local answer.
2. The Fireworks token budgets (800 primary / 1024 retry) were hardcoded
   and too small for minimax-m3's reasoning overhead over a full source
   passage -- observed 3933 reasoning chars (~1000 tokens) consuming the
   entire budget and producing a total failure ("I could not determine a
   reliable answer."). Both budgets are now larger.
"""
from categories.handlers.context import PipelineContext
from categories.handlers.summarization import handle
from categories.metrics import MetricsCollector
from categories.task_categories import SUMMARIZATION
from config import CATEGORY_MAX_TOKENS

TASK = {"task_id": "t1", "prompt": "Summarize this in exactly 5 words: <long passage>."}


class _StubLocalModel:
    def __init__(self, response: str):
        self._response = response

    def is_loaded(self) -> bool:
        return True

    def generate(self, prompt_str: str, **kwargs) -> str:
        return self._response


class NoOpClient:
    def call(self, *args, **kwargs):
        raise AssertionError("Fireworks should not be called: the terse local summary is non-empty")


def test_short_but_nonempty_local_summary_is_accepted():
    # A properly complete short sentence (ends with terminal punctuation,
    # no dangling word) -- run_summarization()'s own completeness check
    # accepts it, and the handler must not additionally reject it for being
    # short.
    short_summary = "Sales grew steadily."
    ctx = PipelineContext(client=NoOpClient(), metrics=MetricsCollector(),
                           local_model=_StubLocalModel(short_summary))

    result = handle(TASK, SUMMARIZATION, ctx)

    assert result == {"task_id": "t1", "answer": short_summary}
    assert ctx.metrics.records[0]["path"] == "local_summarization"


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


class _UnloadedLocalModel:
    def is_loaded(self) -> bool:
        return False


def test_summarization_fireworks_call_uses_the_larger_configured_budget(monkeypatch):
    monkeypatch.setenv("ALLOWED_MODELS", "accounts/fireworks/models/minimax-m3")
    client = _RecordingClient("A concise summary.")
    ctx = PipelineContext(client=client, metrics=MetricsCollector(), local_model=_UnloadedLocalModel())

    handle(TASK, SUMMARIZATION, ctx)

    assert client.calls == [CATEGORY_MAX_TOKENS[SUMMARIZATION]]
    assert CATEGORY_MAX_TOKENS[SUMMARIZATION] > 800  # the old, too-tight budget
