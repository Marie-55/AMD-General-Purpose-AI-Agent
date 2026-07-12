"""
main.py -- container entrypoint.

Loads /input/tasks.json, classifies each prompt into one of 8 fixed
categories, dispatches it to the matching handler in categories/handlers/,
and writes /output/results.json.

Routing philosophy (see categories/handlers/*.py for the concrete per-category
logic and config.CATEGORY_ROLE for the Fireworks role each category maps to):
    sentiment / factual_knowledge / summarization / ner / logic_puzzle
        → local Qwen2.5 first, escalate to Fireworks only if the local
          answer is empty, too short, or otherwise not good enough.
    math_reasoning
        → local Python code generation + execution first (zero tokens);
          Fireworks code generation, then plain NL, as fallbacks.
    logic_puzzle (Fireworks path)
        → reasoning specialist with a truncation-rescue pass.
    code_generation / code_debugging
        → Fireworks code specialist + AST/exec verification + auto-fix retry.

max_tokens is dynamic per category (see config.CATEGORY_MAX_TOKENS).

Prompt/config note: every literal prompt string lives in categories/prompts.py
and every tunable timeout/token-budget/routing table lives in config.py. This
module and categories/handlers/*.py should only ever reference those, never
define new inline literals.
"""
import json
import os
import sys
import time
from concurrent.futures import as_completed

from categories.classifier import classify
from categories.routing import get_allowed_models, resolve_roles
from categories.executor import get_task_executor, get_max_workers, ENV_PROFILE
from categories.metrics import MetricsCollector
from categories.local_model import get_local_model
from categories.handlers import dispatch
from categories.handlers.context import PipelineContext
from config import TOTAL_TIME_BUDGET_S

MAX_WORKERS = get_max_workers()


def process_task(client, task: dict, metrics: MetricsCollector, local_model) -> dict:
    prompt = task["prompt"]
    category = classify(prompt, local_model)
    ctx = PipelineContext(client=client, metrics=metrics, local_model=local_model)
    return dispatch(task, category, ctx)


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    input_path:  str = "/input/tasks.json",
    output_path: str = "/output/results.json",
    client=None,
):
    start = time.time()

    with open(input_path) as f:
        tasks = json.load(f)

    if not isinstance(tasks, list):
        raise ValueError("/input/tasks.json must contain a JSON list")
    for i, task in enumerate(tasks):
        if not isinstance(task, dict) or "task_id" not in task or "prompt" not in task:
            raise ValueError(f"Task at index {i} must contain task_id and prompt")

    get_allowed_models()
    print(f"[startup] Environment profile: {ENV_PROFILE}", file=sys.stderr)

    if client is None:
        from utils.fireworks_client import FireworksClient
        client = FireworksClient()

    # Load local Qwen model once (singleton).
    local_model = None
    try:
        local_model = get_local_model()
        if local_model.is_loaded():
            print("[startup] Local model loaded successfully.", file=sys.stderr)
        else:
            print("[startup] Local model NOT loaded — sentiment/NER will use Fireworks.",
                  file=sys.stderr)
    except Exception as exc:
        print(f"[startup] Local model init failed: {exc} — continuing without it.",
              file=sys.stderr)

    # One-time concurrent health probe of Fireworks models.
    resolved_roles, health = resolve_roles(client)
    print("Role resolution (this run):", file=sys.stderr)
    for role, mdl in resolved_roles.items():
        print(f"  {role} -> {mdl}", file=sys.stderr)
    print("Model health:",
          {m: ("UP" if ok else "DOWN") for m, ok in health.items()},
          file=sys.stderr)

    metrics = MetricsCollector()
    results = [None] * len(tasks)

    with get_task_executor(MAX_WORKERS) as pool:
        futures = {}
        for i, task in enumerate(tasks):
            if time.time() - start > TOTAL_TIME_BUDGET_S:
                results[i] = {"task_id": task["task_id"], "answer": ""}
                continue
            futures[pool.submit(process_task, client, task, metrics, local_model)] = i

        for fut in as_completed(futures):
            i = futures[fut]
            try:
                results[i] = fut.result()
            except Exception as exc:
                print(f"[pipeline] task {tasks[i]['task_id']} raised: {exc}",
                      file=sys.stderr)
                results[i] = {"task_id": tasks[i]["task_id"], "answer": ""}

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    metrics.print_report()
    return results, metrics


if __name__ == "__main__":
    try:
        run_pipeline()
        sys.exit(0)
    except Exception as e:
        print(f"FATAL ERROR: {e}", file=sys.stderr)
        sys.exit(1)
