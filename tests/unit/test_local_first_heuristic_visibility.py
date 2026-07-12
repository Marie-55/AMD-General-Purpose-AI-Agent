"""
Regression tests for the sentiment/NER heuristic fallback: run_sentiment()/
run_ner() tag their answer's source ("llm" vs "heuristic") so callers can
tell a genuine local-model answer from a regex/keyword guess produced only
because the model call failed -- and try_local_first() uses that tag to
NEVER accept a heuristic-sourced answer, always escalating to Fireworks
instead. Before this fix, a heuristic-sourced answer was accepted exactly
like a real one; caught in production by wrong NER labels ("Mountain View"
tagged PERSON) and a self-contradictory sentiment sentence ("positives
(less reliable...) but also negatives (less reliable...)").
"""
import json

from categories.local_model import run_sentiment, run_ner
from categories.handlers.base import try_local_first
from categories.handlers.context import PipelineContext
from categories.metrics import MetricsCollector


class _AlwaysFailingModel:
    def generate(self, *args, **kwargs):
        raise TimeoutError("Local model busy: timed out after 8.0s waiting for a turn.")


class _WorkingSentimentModel:
    def generate(self, prompt_str, **kwargs):
        return "positive: great product."


class _WorkingNerModel:
    def generate(self, prompt_str, **kwargs):
        return '[{"text": "Paris", "type": "LOCATION"}]'


def test_run_sentiment_reports_heuristic_source_when_model_call_fails():
    answer, source = run_sentiment(
        "The product is great but the box was damaged.", _AlwaysFailingModel()
    )
    assert source == "heuristic"
    assert answer.strip()


def test_run_sentiment_reports_llm_source_when_model_call_succeeds():
    answer, source = run_sentiment("This product is absolutely great.", _WorkingSentimentModel())
    assert source == "llm"


class _NeutralModel:
    """A model call that succeeds (no exception, parseable label) but gets
    overridden by the heuristic's own verdict -- source must still be
    "heuristic" here, since the *returned text* is the heuristic's, not the
    model's own words."""

    def generate(self, prompt_str, **kwargs):
        return "neutral"


def test_run_sentiment_reports_heuristic_source_when_model_call_succeeds_but_is_overridden():
    # Regression guard: caught in a real run producing a grammatically
    # broken justification ("it has positives (is fortunate because the)
    # but also negatives (charger stopped working on day).") that was
    # accepted as an ordinary local success because source was computed
    # once, up front, from "did the model call succeed" -- not from which
    # text (model's vs. heuristic's) was actually returned.
    text = "I am fortunate because the battery lasts all day, but the charger stopped working on day three."
    answer, source = run_sentiment(text, _NeutralModel())
    assert source == "heuristic"
    assert not answer.lower().startswith("neutral")  # the heuristic overrode the model's "neutral"


def test_run_ner_reports_heuristic_source_when_model_call_fails():
    answer, source = run_ner(
        "Extract entities: Barack Obama visited Paris in 2011.", _AlwaysFailingModel()
    )
    assert source == "heuristic"
    parsed = json.loads(answer)
    assert isinstance(parsed, list)
    assert parsed  # the heuristic should find at least one capitalized entity


def test_run_ner_reports_llm_source_when_model_call_succeeds():
    answer, source = run_ner("Extract entities: I visited Paris.", _WorkingNerModel())
    assert source == "llm"
    assert "Paris" in answer


class _StubLocalModel:
    def is_loaded(self):
        return True


def test_try_local_first_never_accepts_a_heuristic_sourced_answer():
    ctx = PipelineContext(client=None, metrics=MetricsCollector(), local_model=_StubLocalModel())
    result = try_local_first(
        ctx, "t1", "sentiment", "some prompt",
        # is_good_enough would happily accept this -- the point of the test
        # is that source == "heuristic" must override that entirely.
        local_fn=lambda prompt, model: ("heuristic answer", "heuristic"),
        is_good_enough=lambda a: True,
        path_name="local_sentiment",
    )
    assert result is None
    assert ctx.metrics.records == []


def test_try_local_first_leaves_llm_source_path_name_unchanged():
    ctx = PipelineContext(client=None, metrics=MetricsCollector(), local_model=_StubLocalModel())
    result = try_local_first(
        ctx, "t1", "sentiment", "some prompt",
        local_fn=lambda prompt, model: ("model answer", "llm"),
        is_good_enough=lambda a: bool(a),
        path_name="local_sentiment",
    )
    assert result == "model answer"
    assert ctx.metrics.records[0]["path"] == "local_sentiment"


def test_try_local_first_treats_plain_string_as_llm_source():
    ctx = PipelineContext(client=None, metrics=MetricsCollector(), local_model=_StubLocalModel())
    result = try_local_first(
        ctx, "t1", "factual_knowledge", "some prompt",
        local_fn=lambda prompt, model: "plain string answer",
        is_good_enough=lambda a: bool(a),
        path_name="local_factual",
    )
    assert result == "plain string answer"
    assert ctx.metrics.records[0]["path"] == "local_factual"
