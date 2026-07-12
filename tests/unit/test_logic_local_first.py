"""
Phase 5 behavior-change coverage: logic_puzzle now tries the local model
first (mirroring sentiment/summarization/factual/ner) and only escalates to
Fireworks if the local answer fails the confidence gate (see
categories/handlers/logic.py::_logic_answer_is_confident).
"""
from categories.handlers.context import PipelineContext
from categories.handlers.logic import handle
from categories.metrics import MetricsCollector
from categories.task_categories import LOGIC_PUZZLE

TASK = {"task_id": "t1", "prompt": "Logic puzzle: Alice sits left of Bob. Who sits where?"}


class StubLocalModel:
    def __init__(self, response: str):
        self._response = response

    def is_loaded(self) -> bool:
        return True

    def generate(self, prompt_str: str, **kwargs) -> str:
        return self._response


class NoOpClient:
    def call(self, *args, **kwargs):
        raise AssertionError("Fireworks should not be called: the local logic answer is confident")


class StubFireworksClient:
    def __init__(self, answer: str):
        self._answer = answer
        self.called = False

    def call(self, model, messages, max_tokens=400, timeout=25, **kwargs):
        self.called = True
        lat = {
            "model": model, "ttft_ms": 1.0, "total_latency_ms": 2.0,
            "prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10, "error": None,
        }
        return self._answer, lat


def test_logic_accepts_confident_local_answer_without_calling_fireworks():
    confident = (
        "Alice must sit to the left of Bob based on the clue given, and since "
        "there are only two seats, Bob sits to the right.\n\n"
        "Answer: Alice is on the left, Bob is on the right."
    )
    ctx = PipelineContext(client=NoOpClient(), metrics=MetricsCollector(),
                           local_model=StubLocalModel(confident))

    result = handle(TASK, LOGIC_PUZZLE, ctx)

    assert result == {"task_id": "t1", "answer": confident}
    assert ctx.metrics.records[0]["path"] == "local_logic"


def test_logic_escalates_when_local_answer_has_no_conclusion(monkeypatch):
    monkeypatch.setenv("ALLOWED_MODELS", "accounts/fireworks/models/minimax-m3")
    incomplete = "Alice sits somewhere and Bob sits somewhere else, thinking through the clues here"
    fw_answer = "Working through the clues.\n\nAnswer: Alice is on the left, Bob is on the right."
    fw_client = StubFireworksClient(fw_answer)
    ctx = PipelineContext(client=fw_client, metrics=MetricsCollector(),
                           local_model=StubLocalModel(incomplete))

    result = handle(TASK, LOGIC_PUZZLE, ctx)

    assert fw_client.called
    assert result == {"task_id": "t1", "answer": fw_answer}
    assert ctx.metrics.records[0]["path"] == "logic_nl"


def test_logic_escalates_when_local_model_not_loaded(monkeypatch):
    monkeypatch.setenv("ALLOWED_MODELS", "accounts/fireworks/models/minimax-m3")

    class UnloadedModel(StubLocalModel):
        def is_loaded(self):
            return False

    fw_answer = "Reasoning here.\n\nAnswer: Alice left, Bob right."
    fw_client = StubFireworksClient(fw_answer)
    ctx = PipelineContext(client=fw_client, metrics=MetricsCollector(),
                           local_model=UnloadedModel("irrelevant"))

    result = handle(TASK, LOGIC_PUZZLE, ctx)

    assert fw_client.called
    assert result == {"task_id": "t1", "answer": fw_answer}
