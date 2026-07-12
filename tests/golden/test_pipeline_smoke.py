"""
Golden smoke test: runs the real run_pipeline() end-to-end against a fixed
task set with a fake Fireworks client (no network, no API key) and a local
model that fails to load (paths don't exist in the dev/CI environment,
exactly like this sandbox). Asserts every category still produces a
well-formed, non-empty answer of the expected shape.

This is the regression harness the refactor plan checks after every phase --
it must keep passing unchanged except where a phase note says otherwise.
"""
import json
from pathlib import Path

import pytest

from main import run_pipeline
from tests.golden.fake_client import FakeFireworksClient

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "tasks_golden.json"

ALLOWED_MODELS = (
    "accounts/fireworks/models/gemma-4-26b-a4b-it,"
    "accounts/fireworks/models/minimax-m3,"
    "accounts/fireworks/models/kimi-k2p7-code,"
    "accounts/fireworks/models/gemma-4-31b-it,"
    "accounts/fireworks/models/gemma-4-31b-it-nvfp4"
)


@pytest.fixture
def golden_results(tmp_path, monkeypatch):
    monkeypatch.setenv("ALLOWED_MODELS", ALLOWED_MODELS)
    # Force the local-model lookup to miss so every category exercises its
    # Fireworks escalation path deterministically (matches this sandbox,
    # where llama-cpp-python / the .gguf files are not present).
    monkeypatch.setenv("LOCAL_MODEL_PATH", "/nonexistent/primary.gguf")
    monkeypatch.setenv("LOCAL_MODEL_FALLBACK_PATH", "/nonexistent/fallback.gguf")

    import categories.local_model as local_model_module
    monkeypatch.setattr(local_model_module, "_instance", None)

    output_path = tmp_path / "results.json"
    client = FakeFireworksClient()
    results, metrics = run_pipeline(
        input_path=str(FIXTURE_PATH), output_path=str(output_path), client=client
    )
    on_disk = json.loads(output_path.read_text())
    return {"results": results, "on_disk": on_disk, "metrics": metrics}


def _by_task_id(results):
    return {r["task_id"]: r["answer"] for r in results}


def test_writes_one_result_per_task_with_matching_ids(golden_results):
    results = golden_results["results"]
    fixture_tasks = json.loads(FIXTURE_PATH.read_text())
    assert {r["task_id"] for r in results} == {t["task_id"] for t in fixture_tasks}
    assert golden_results["on_disk"] == results


def test_every_answer_is_a_non_empty_string(golden_results):
    for r in golden_results["results"]:
        assert isinstance(r["answer"], str)
        assert r["answer"].strip()


def test_factual_answer_is_not_a_json_blob(golden_results):
    # Regression guard for the factual_knowledge / NER mix-up bug.
    answer = _by_task_id(golden_results["results"])["t1"]
    assert not answer.strip().startswith("[")
    assert "rainbow" in answer.lower() or "droplet" in answer.lower()


def test_ner_answer_is_valid_json_array(golden_results):
    answer = _by_task_id(golden_results["results"])["t5"]
    parsed = json.loads(answer)
    assert isinstance(parsed, list)
    assert all("text" in item and "type" in item for item in parsed)


def test_math_answer_is_present(golden_results):
    answer = _by_task_id(golden_results["results"])["t2"]
    assert answer.strip()


def test_sentiment_answer_starts_with_a_label(golden_results):
    answer = _by_task_id(golden_results["results"])["t3"].lower()
    assert answer.startswith(("positive", "negative", "neutral", "mixed"))


def test_logic_answer_has_a_final_answer_line(golden_results):
    answer = _by_task_id(golden_results["results"])["t6"]
    assert "answer:" in answer.lower()


def test_code_debugging_answer_has_corrected_code_section(golden_results):
    answer = _by_task_id(golden_results["results"])["t7"]
    assert "corrected code" in answer.lower() or "```" in answer


def test_code_generation_answer_contains_code(golden_results):
    answer = _by_task_id(golden_results["results"])["t8"]
    assert "```" in answer or "def " in answer
