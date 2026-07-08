"""
main.py -- container entrypoint.

Reads /input/tasks.json, classifies + normalizes + routes each task, executes
via Fireworks (with a local code-gen/exec shortcut for math & logic), and
writes /output/results.json. Exits 0 on success, non-zero on failure, and
enforces the runtime/per-request time budgets from the rules.

This file is intentionally thin: all logic lives in categories/ and utils/ so
that the SAME code is exercised by the dev notebook and by the submitted
Docker image.
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from categories.classifier import classify
from categories.normalizer import normalize_prompt
from categories.routing import route, CODE_EXEC_CATEGORIES, get_allowed_models, resolve_roles
from categories.code_exec import build_codegen_messages, extract_code, run_code_safely
from categories.metrics import MetricsCollector

# Hard limits from the rules: 10 min total runtime, 30s per request, 60s startup.
# We budget under those with safety margin.
TOTAL_TIME_BUDGET_S = 9 * 60
PER_REQUEST_TIMEOUT_S = 25
CODE_EXEC_TIMEOUT_S = 8.0
MAX_WORKERS = 6


def process_task(client, task, metrics: MetricsCollector):
    task_id = task["task_id"]
    prompt = task["prompt"]
    category = classify(prompt)

    if category in CODE_EXEC_CATEGORIES:
        answer, path, model, lat = _solve_via_code(client, prompt, category)
    else:
        model = route(category)
        messages = normalize_prompt(prompt, category)
        answer, lat = client.call(model, messages, max_tokens=350, timeout=PER_REQUEST_TIMEOUT_S)
        path = "direct_llm"
        if lat.get("error"):
            answer = answer or ""

    metrics.log(task_id, category, path, model, lat)
    return {"task_id": task_id, "answer": answer}


def _solve_via_code(client, prompt, category):
    model = route(category)
    messages = build_codegen_messages(prompt)
    llm_out, lat = client.call(model, messages, max_tokens=500, timeout=PER_REQUEST_TIMEOUT_S)

    if lat.get("error") or not llm_out:
        fallback_answer, fallback_lat = _direct_fallback_call(client, prompt, category, model)
        return fallback_answer, "code_exec_fallback", model, fallback_lat

    code = extract_code(llm_out)
    ok, output = run_code_safely(code, timeout_s=CODE_EXEC_TIMEOUT_S)
    if ok and output:
        return output, "code_exec", model, lat

    fallback_answer, fallback_lat = _direct_fallback_call(client, prompt, category, model)
    return fallback_answer, "code_exec_fallback", model, fallback_lat


def _direct_fallback_call(client, prompt, category, model):
    messages = normalize_prompt(prompt, category)
    return client.call(model, messages, max_tokens=350, timeout=PER_REQUEST_TIMEOUT_S)


def run_pipeline(input_path="/input/tasks.json", output_path="/output/results.json", client=None):
    start = time.time()

    with open(input_path) as f:
        tasks = json.load(f)

    if not isinstance(tasks, list):
        raise ValueError("/input/tasks.json must contain a JSON list")

    for i, task in enumerate(tasks):
        if not isinstance(task, dict) or "task_id" not in task or "prompt" not in task:
            raise ValueError(f"Task at index {i} must contain task_id and prompt")

    # Validate model configuration before dispatching worker threads.
    # If ALLOWED_MODELS is missing, this is a configuration failure and should
    # fail clearly instead of producing empty answers.
    get_allowed_models()

    if client is None:
        from utils.fireworks_client import FireworksClient
        client = FireworksClient()

    # One-time cascade: probe every candidate model concurrently, cache
    # which ones actually respond, and route the rest of the run off that.
    # Cheap (max_tokens=1 per probe) and fast (parallel, not sequential).
    resolved_roles, health = resolve_roles(client)
    print("Role resolution (this run):", file=sys.stderr)
    for role, model in resolved_roles.items():
        print(f"  {role} -> {model}", file=sys.stderr)
    print("Model health:", {m: ("UP" if ok else "DOWN") for m, ok in health.items()}, file=sys.stderr)

    metrics = MetricsCollector()
    results = [None] * len(tasks)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {}

        for i, task in enumerate(tasks):
            if time.time() - start > TOTAL_TIME_BUDGET_S:
                # Out of time budget: emit an empty answer rather than risk
                # exceeding the hard 10-minute runtime limit.
                results[i] = {"task_id": task["task_id"], "answer": ""}
                continue

            futures[pool.submit(process_task, client, task, metrics)] = i

        for fut in as_completed(futures):
            i = futures[fut]
            try:
                results[i] = fut.result()
            except Exception:
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