"""Container entrypoint -- fully local agent, zero Fireworks tokens.

Reads /input/tasks.json (falls back to ./input/tasks.json for local runs),
classifies each task into one of the 8 competition categories, groups tasks
by which of the two local models they need so each model is loaded at most
once, runs every task, and writes /output/results.json (falls back to
./output/results.json).

Evaluation metrics are printed to stdout and written to
/output/metrics.json so you can inspect them after a run.
"""
import json
import time
from pathlib import Path

import config
from categories.classifier import classify
from categories.handlers import DISPATCH
from categories.model_runtime import ModelRuntime

APP_DIR = Path(__file__).resolve().parent


def _resolve_input_path() -> Path:
    container_path = Path("/input/tasks.json")
    if container_path.exists():
        return container_path
    return APP_DIR / "input" / "tasks.json"


def _resolve_output_path() -> Path:
    container_dir = Path("/output")
    try:
        container_dir.mkdir(exist_ok=True)
        return container_dir / "results.json"
    except (PermissionError, OSError):
        local_dir = APP_DIR / "output"
        local_dir.mkdir(exist_ok=True)
        return local_dir / "results.json"


def _resolve_metrics_path() -> Path:
    results_path = _resolve_output_path()
    return results_path.parent / "metrics.json"


def load_tasks() -> list:
    path = _resolve_input_path()
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_results(results: list) -> None:
    path = _resolve_output_path()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(results)} results to {path}", flush=True)


def write_metrics(metrics: dict) -> None:
    path = _resolve_metrics_path()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Wrote metrics to {path}", flush=True)


def _count_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars/token (good enough for local tracking)."""
    return max(1, len(text) // 4)


def run_task(task: dict, category: str, runtime: ModelRuntime) -> tuple[str, dict]:
    """Returns (answer, per_task_metrics)."""
    t0 = time.monotonic()
    handler = DISPATCH[category]
    try:
        answer = handler(task, runtime)
    except Exception as exc:
        print(f"[main] task {task.get('task_id')} ({category}) failed: {exc}", flush=True)
        answer = config.FALLBACK_ANSWER

    elapsed = time.monotonic() - t0
    answer = answer or config.FALLBACK_ANSWER

    # Estimate token usage (local = 0 Fireworks tokens, but useful for perf analysis)
    prompt_tokens = _count_tokens(task["prompt"])
    output_tokens = _count_tokens(answer)
    is_fallback = answer == config.FALLBACK_ANSWER

    task_metrics = {
        "task_id": task["task_id"],
        "category": category,
        "elapsed_s": round(elapsed, 3),
        "prompt_tokens_est": prompt_tokens,
        "output_tokens_est": output_tokens,
        "total_tokens_est": prompt_tokens + output_tokens,
        "is_fallback": is_fallback,
        "answer_length": len(answer),
        "fireworks_tokens": 0,  # fully local -- always 0
    }

    status = "FALLBACK" if is_fallback else "OK"
    timeout_warn = " ⚠ SLOW" if elapsed > config.PER_TASK_SOFT_LIMIT_S else ""
    print(
        f"  [{status}] {task['task_id']} ({category}) "
        f"in {elapsed:.2f}s | ~{output_tokens} out-tokens{timeout_warn}",
        flush=True,
    )

    return answer, task_metrics


def _print_summary(task_metrics_list: list, total_elapsed: float) -> dict:
    """Print a structured summary and return the metrics dict."""
    n = len(task_metrics_list)
    fallbacks = sum(1 for m in task_metrics_list if m["is_fallback"])
    answered = n - fallbacks
    slow_tasks = [m for m in task_metrics_list if m["elapsed_s"] > config.PER_TASK_SOFT_LIMIT_S]
    total_out_tokens = sum(m["output_tokens_est"] for m in task_metrics_list)
    total_in_tokens = sum(m["prompt_tokens_est"] for m in task_metrics_list)

    # Tasks answered (not fallback) / total -- proxy for accuracy
    answer_rate = answered / n if n else 0.0
    # Competition uses 19 hidden tasks. If this run has 19, estimate is direct.
    # For 8-task dev runs, we scale to give an indicative number only.
    estimated_gate = "LIKELY PASS" if answer_rate >= 0.80 else "AT RISK"

    lines = [
        "",
        "=" * 60,
        "EVALUATION METRICS SUMMARY",
        "=" * 60,
        f"  Total tasks          : {n}",
        f"  Answered (non-fb)    : {answered}",
        f"  Fallback responses   : {fallbacks}",
        f"  Answer rate          : {answer_rate:.1%}  ({estimated_gate} for 80% gate)",
        f"  Total runtime        : {total_elapsed:.1f}s  (budget: {config.TOTAL_TIME_BUDGET_S}s)",
        f"  Budget used          : {total_elapsed / config.TOTAL_TIME_BUDGET_S:.1%}",
        f"  Fireworks tokens     : 0  (fully local run)",
        f"  Est. input tokens    : {total_in_tokens}",
        f"  Est. output tokens   : {total_out_tokens}",
        f"  Est. total tokens    : {total_in_tokens + total_out_tokens}",
    ]
    if slow_tasks:
        lines.append(f"  Slow tasks (>{config.PER_TASK_SOFT_LIMIT_S}s) : "
                     + ", ".join(m["task_id"] for m in slow_tasks))
    lines.append("")
    lines.append("  PER-TASK BREAKDOWN:")
    lines.append(f"  {'ID':<8} {'Category':<22} {'Time':>7}  {'OutTok':>7}  Status")
    lines.append("  " + "-" * 55)
    for m in task_metrics_list:
        flag = "⚠ SLOW" if m["elapsed_s"] > config.PER_TASK_SOFT_LIMIT_S else ""
        fb = "FALLBACK" if m["is_fallback"] else "ok"
        lines.append(
            f"  {m['task_id']:<8} {m['category']:<22} {m['elapsed_s']:>6.2f}s  "
            f"{m['output_tokens_est']:>6}t  {fb} {flag}"
        )
    lines.append("=" * 60)

    for line in lines:
        print(line, flush=True)

    metrics = {
        "run_summary": {
            "total_tasks": n,
            "answered": answered,
            "fallbacks": fallbacks,
            "answer_rate": round(answer_rate, 4),
            "accuracy_gate_estimate": estimated_gate,
            "total_elapsed_s": round(total_elapsed, 2),
            "budget_s": config.TOTAL_TIME_BUDGET_S,
            "budget_used_pct": round(total_elapsed / config.TOTAL_TIME_BUDGET_S * 100, 1),
            "fireworks_tokens": 0,
            "estimated_input_tokens": total_in_tokens,
            "estimated_output_tokens": total_out_tokens,
            "estimated_total_tokens": total_in_tokens + total_out_tokens,
            "slow_tasks": [m["task_id"] for m in slow_tasks],
        },
        "per_task": task_metrics_list,
    }
    return metrics


def main() -> None:
    start = time.monotonic()
    tasks = load_tasks()
    print(f"Loaded {len(tasks)} tasks", flush=True)

    categorized = [(task, classify(task["prompt"])) for task in tasks]
    for task, category in categorized:
        print(f"  {task['task_id']}: {category}", flush=True)

    buckets = {"coder": [], "generalist": []}
    for task, category in categorized:
        buckets[config.CATEGORY_MODEL[category]].append((task, category))

    answers = {}
    task_metrics_list = []
    runtime = ModelRuntime()

    # Process the coder bucket first, then the generalist bucket -- each
    # model is loaded exactly once (or not at all if its bucket is empty).
    for model_key in ("coder", "generalist"):
        group = buckets[model_key]
        if not group:
            continue
        print(f"\nLoading {model_key} model for {len(group)} task(s)...", flush=True)
        try:
            runtime.load(model_key)
        except Exception as exc:
            print(f"[main] FATAL: could not load {model_key} model: {exc}", flush=True)
            for task, category in group:
                answers[task["task_id"]] = config.FALLBACK_ANSWER
                task_metrics_list.append({
                    "task_id": task["task_id"],
                    "category": category,
                    "elapsed_s": 0.0,
                    "prompt_tokens_est": _count_tokens(task["prompt"]),
                    "output_tokens_est": 0,
                    "total_tokens_est": _count_tokens(task["prompt"]),
                    "is_fallback": True,
                    "answer_length": len(config.FALLBACK_ANSWER),
                    "fireworks_tokens": 0,
                    "error": str(exc),
                })
            continue

        for task, category in group:
            if time.monotonic() - start > config.TOTAL_TIME_BUDGET_S:
                print("[main] time budget exhausted, using fallback for remaining tasks", flush=True)
                answers[task["task_id"]] = config.FALLBACK_ANSWER
                task_metrics_list.append({
                    "task_id": task["task_id"],
                    "category": category,
                    "elapsed_s": 0.0,
                    "prompt_tokens_est": _count_tokens(task["prompt"]),
                    "output_tokens_est": 0,
                    "total_tokens_est": _count_tokens(task["prompt"]),
                    "is_fallback": True,
                    "answer_length": len(config.FALLBACK_ANSWER),
                    "fireworks_tokens": 0,
                    "error": "budget_exhausted",
                })
                continue
            answer, tm = run_task(task, category, runtime)
            answers[task["task_id"]] = answer
            task_metrics_list.append(tm)

    runtime.unload()

    results = [
        {"task_id": task["task_id"], "answer": answers[task["task_id"]]}
        for task, _ in categorized
    ]
    write_results(results)

    total_elapsed = time.monotonic() - start
    metrics = _print_summary(task_metrics_list, total_elapsed)
    write_metrics(metrics)

    print(f"\nDone in {total_elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
