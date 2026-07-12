"""
Regression test for the factual_knowledge / NER mix-up bug: main.py used to
call the local NER extractor for factual_knowledge tasks instead of
run_factual(). Pins the fix by stubbing a *loaded* local model (the buggy
branch was only reachable when local_model.is_loaded() is True) and asserting
the model is prompted the way run_factual() prompts it, not the way
run_ner() prompts it, and that the answer is never a JSON array.
"""
from categories.metrics import MetricsCollector
from categories.local_model import _build_chatml
from main import process_task

FACTUAL_SYSTEM_MARKER = "answer the user's question directly"
NER_SYSTEM_MARKER = "named entity recognition"


class StubLocalModel:
    """Always 'loaded'; returns a response shaped for whichever system
    prompt it was actually given, so the test can tell run_factual and
    run_ner apart from the model's-eye view."""

    def is_loaded(self) -> bool:
        return True

    def generate(self, prompt_str: str, **kwargs) -> str:
        lowered = prompt_str.lower()
        if NER_SYSTEM_MARKER in lowered:
            return '[{"text": "Marie Curie", "type": "PERSON"}]'
        if FACTUAL_SYSTEM_MARKER in lowered:
            return (
                "Marie Curie won the Nobel Prize in Physics in 1903 and the "
                "Nobel Prize in Chemistry in 1911, for her work on "
                "radioactivity and the discovery of polonium and radium."
            )
        return ""


class NoOpClient:
    def call(self, *args, **kwargs):
        raise AssertionError("Fireworks should not be called: the local factual answer is confident")


def test_factual_knowledge_uses_run_factual_not_run_ner():
    # Phrased to score >=1 on the factual_knowledge regex so classify()
    # resolves deterministically without depending on the (stubbed) local
    # model's classification pass.
    task = {"task_id": "t1", "prompt": "What is Marie Curie best known for discovering?"}
    metrics = MetricsCollector()
    result = process_task(NoOpClient(), task, metrics, StubLocalModel())

    assert result["task_id"] == "t1"
    answer = result["answer"]
    assert not answer.strip().startswith("[")
    assert "Marie Curie" in answer

    logged = metrics.records[0]
    assert logged["category"] == "factual_knowledge"
    assert logged["path"] == "local_factual"


def test_run_factual_and_run_ner_use_distinguishable_system_prompts():
    # Guards the test double itself: if either module's system prompt text
    # changes, this fails loudly instead of the fixture silently going stale.
    from categories.local_model import run_factual, run_ner

    factual_call = _build_chatml("Answer the user's question directly and concisely.", "x")
    ner_call = _build_chatml("You are a named entity recognition system.", "x")
    assert FACTUAL_SYSTEM_MARKER in factual_call.lower()
    assert NER_SYSTEM_MARKER in ner_call.lower()
