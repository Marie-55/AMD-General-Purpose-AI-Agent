from __future__ import annotations

from pathlib import Path

from categories.router import TaskRouter
from utils.agent import TrackOneAgent
from utils.heuristics import classify_heuristically, pick_model
from utils.models import RuntimeConfig, Task, TaskCategory
from utils.parsing import parse_json_loose
from utils.sandbox import ExecutionSandbox
from utils.task_io import read_tasks, write_results


class FakeClient:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls = []

    def chat_completion(self, model, messages, *, temperature=0.0, max_tokens=None, top_p=None, endpoint="chat/completions"):
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "endpoint": endpoint,
            }
        )
        text = self.responses.pop(0)
        return text, (3, 2), {"choices": [{"message": {"content": text}}], "usage": {"prompt_tokens": 3, "completion_tokens": 2}}


class FallbackClient:
    def __init__(self):
        self.calls = []

    def chat_completion(self, model, messages, *, temperature=0.0, max_tokens=None, top_p=None, endpoint="chat/completions"):
        self.calls.append(model)
        if "gemma" in model:
            raise RuntimeError("404 model not found")
        text = '{"category":"factual","confidence":0.9,"reason":"fallback"}' if any("available categories" in m.get("content", "") for m in messages) else "Paris"
        return text, (4, 1), {"choices": [{"message": {"content": text}}], "usage": {"prompt_tokens": 4, "completion_tokens": 1}}


def make_config(tmp_path: Path) -> RuntimeConfig:
    return RuntimeConfig(
        api_key="test-key",
        base_url="https://example.com",
        allowed_models=[
            "minimax-m3",
            "kimi-k2p7-code",
            "gemma-4-31b-it",
        ],
        input_path=tmp_path / "tasks.json",
        output_path=tmp_path / "results.json",
    )


def test_heuristic_router_detects_summary():
    decision = classify_heuristically("Summarise the following text in one sentence.")
    assert decision.category == TaskCategory.SUMMARY.value


def test_json_parser_handles_fences():
    payload = parse_json_loose("```json\n{\"label\":\"positive\",\"reason\":\"clear praise\"}\n```")
    assert payload["label"] == "positive"


def test_sandbox_prints_answer_variable():
    sandbox = ExecutionSandbox(timeout_s=5)
    result = sandbox.run("answer = 2 + 2")
    assert result.returncode == 0
    assert result.stdout == "4"


def test_model_picker_prefers_low_cost_model():
    config = RuntimeConfig(
        api_key="k",
        base_url="https://example.com",
        allowed_models=["gemma-4-31b-it", "kimi-k2p7-code", "minimax-m3"],
    )
    assert pick_model(config, "router") == "minimax-m3"
    assert pick_model(config, "code") == "kimi-k2p7-code"


def test_agent_runs_with_fake_client(tmp_path):
    config = make_config(tmp_path)
    fake_client = FakeClient(
        [
            '{"category":"factual","confidence":0.98,"reason":"clear factual question"}',
            "The Eiffel Tower is in Paris.",
        ]
    )
    agent = TrackOneAgent(config, client=fake_client)
    result = agent.solve_task(Task(task_id="t1", prompt="Where is the Eiffel Tower?"))
    assert result.task_id == "t1"
    assert "Paris" in result.answer


def test_model_fallback_skips_unavailable_gemma(tmp_path):
    config = RuntimeConfig(
        api_key="k",
        base_url="https://example.com",
        allowed_models=["gemma-4-31b-it", "kimi-k2p7-code", "minimax-m3"],
        input_path=tmp_path / "tasks.json",
        output_path=tmp_path / "results.json",
    )
    agent = TrackOneAgent(config, client=FallbackClient())
    result = agent.solve_task(Task(task_id="t1", prompt="Explain what the CAP theorem states."))
    assert result.task_id == "t1"


def test_task_io_roundtrip(tmp_path):
    path = tmp_path / "tasks.json"
    path.write_text('[{"task_id":"t1","prompt":"hello"}]', encoding="utf-8")
    tasks = read_tasks(path)
    assert tasks[0].task_id == "t1"
    out = tmp_path / "results.json"
    write_results(out, [])
    assert out.read_text(encoding="utf-8").strip() == "[]"
