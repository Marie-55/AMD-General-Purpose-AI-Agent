"""
main.py -- container entrypoint.

Routing logic:
  sentiment / NER      → local Qwen2.5-1.5B (zero Fireworks tokens)
  math_reasoning       → local Python execution first (zero tokens);
                         Fireworks NL fallback only if local exec fails
  logic_puzzle         → Fireworks NL directly (reasoning specialist);
                         code-exec skipped — constraint deduction as Python
                         is unreliable
  code_gen / debug     → Fireworks code specialist + AST/exec verification
  factual / summary    → Fireworks cheap_general with prompt compression

max_tokens is dynamic per category (see normalizer.CATEGORY_MAX_TOKENS).
Summarization answers with exact word-count constraints are post-processed.
"""
import json
import os
import re
import sys
import time
from concurrent.futures import as_completed

from categories.classifier import classify
from categories.normalizer import (
    normalize_prompt, get_max_tokens, enforce_summarization_constraint
)
from categories.routing import (
    route, CODE_EXEC_CATEGORIES, LOGIC_NL_CATEGORIES,
    get_allowed_models, resolve_roles,
)
from categories.code_exec import (
    build_codegen_messages, extract_code, run_code_safely,
)
from categories.code_verifier import verify_and_fix
from categories.prompt_compressor import compress_messages, estimate_tokens
from categories.executor import get_task_executor, get_max_workers, ENV_PROFILE
from categories.metrics import MetricsCollector
from categories.local_model import get_local_model, run_sentiment, run_ner

# Hard limits from the competition rules.
TOTAL_TIME_BUDGET_S   = 9 * 60
PER_REQUEST_TIMEOUT_S = 25
CODE_EXEC_TIMEOUT_S   = 8.0
MAX_WORKERS           = get_max_workers()

# Categories answered entirely by the local Qwen model — zero Fireworks tokens.
LOCAL_CATEGORIES = {"sentiment", "ner"}

# Categories routed through code-gen + AST/exec verification.
CODE_VERIFY_CATEGORIES = {"code_generation", "code_debugging"}

# Prompt compression threshold: only compress user messages longer than this.
COMPRESS_THRESHOLD = 300   # words


# ---------------------------------------------------------------------------
# Per-task processing
# ---------------------------------------------------------------------------

def process_task(client, task: dict, metrics: MetricsCollector, local_model) -> dict:
    task_id  = task["task_id"]
    prompt   = task["prompt"]
    category = classify(prompt, local_model)

    # ── Local model path (sentiment / NER) ─────────────────────────────────
    if category in LOCAL_CATEGORIES and local_model is not None and local_model.is_loaded():
        t0 = time.time()
        try:
            answer = (run_sentiment(prompt, local_model) if category == "sentiment"
                      else run_ner(prompt, local_model))
        except Exception as exc:
            print(f"[local] {task_id} {category} failed: {exc}", file=sys.stderr)
            answer = ""
        elapsed_ms = round((time.time() - t0) * 1000, 1)

        # NER guard: empty result likely means it's not a real NER task.
        if category == "ner" and (not answer or answer.strip() in ("", "[]", "[ ]")):
            print(f"[local] {task_id} ner: empty result, re-routing to factual_knowledge",
                  file=sys.stderr)
            category = "factual_knowledge"
            # Fall through to Fireworks below.
        elif answer:
            lat = {
                "total_latency_ms": elapsed_ms,
                "ttft_ms": None, "prompt_tokens": None,
                "completion_tokens": None, "total_tokens": None, "error": None,
            }
            metrics.log(task_id, category, f"local_{category}", "local_qwen2.5", lat)
            return {"task_id": task_id, "answer": answer}
        else:
            print(f"[local] {task_id} {category}: empty, falling back to Fireworks",
                  file=sys.stderr)

    # ── Math: local Python execution first (zero Fireworks tokens) ──────────
    if category in CODE_EXEC_CATEGORIES:           # {"math_reasoning"}
        answer, path, model, lat = _solve_math(client, prompt, category)
        metrics.log(task_id, category, path, model, lat)
        return {"task_id": task_id, "answer": answer}

    # ── Logic: straight to Fireworks NL (no code-exec attempt) ─────────────
    if category in LOGIC_NL_CATEGORIES:            # {"logic_puzzle"}
        answer, path, model, lat = _solve_logic_nl(client, prompt)
        metrics.log(task_id, category, path, model, lat)
        return {"task_id": task_id, "answer": answer}

    # ── Code generation / debugging with verification ───────────────────────
    if category in CODE_VERIFY_CATEGORIES:
        answer, path, model, lat = _solve_via_code_verify(client, prompt, category)
        metrics.log(task_id, category, path, model, lat)
        return {"task_id": task_id, "answer": answer}

    # ── General Fireworks path (factual, summarization, NER fallback, …) ───
    model    = route(category)
    messages = normalize_prompt(prompt, category)
    messages = _maybe_compress(messages, prompt)
    max_tok  = get_max_tokens(category)
    answer, lat = client.call(model, messages, max_tokens=max_tok,
                              timeout=PER_REQUEST_TIMEOUT_S)
    answer = answer or ""

    # Post-process summarization word-count constraints.
    if category == "summarization" and answer:
        answer = enforce_summarization_constraint(answer, prompt)

    if lat.get("error"):
        answer = answer or ""
    metrics.log(task_id, category, "direct_llm", model, lat)
    return {"task_id": task_id, "answer": answer}


# ---------------------------------------------------------------------------
# Math: local execution (zero Fireworks tokens) → Fireworks NL fallback
# ---------------------------------------------------------------------------

def _solve_math(client, prompt: str, category: str):
    """
    Try to answer a math problem by generating + running Python locally.

    Step 1: Ask Fireworks code_specialist for a short Python script.
            (One Fireworks call — but only the script, not the full answer.)
    Step 2: Execute the script locally — zero extra tokens.
    Step 3: If exec fails/errors, fall back to a plain NL Fireworks call.

    Net result: on success, we spend tokens only on a short script generation
    (much cheaper than a full chain-of-thought answer). On failure we spend
    tokens on an NL answer — same as the old path but with a better chance of
    success because we tried local exec first.
    """
    model        = route(category)
    messages     = build_codegen_messages(prompt)
    llm_out, lat = client.call(model, messages, max_tokens=400,
                               timeout=PER_REQUEST_TIMEOUT_S)

    if not lat.get("error") and llm_out:
        code = extract_code(llm_out)
        ok, output = run_code_safely(code, timeout_s=CODE_EXEC_TIMEOUT_S)
        if ok and output:
            print(f"[math] local exec succeeded: {output[:60]}", file=sys.stderr)
            return output, "math_local_exec", model, lat

    # Local exec failed — fall back to NL answer.
    print(f"[math] local exec failed, falling back to NL", file=sys.stderr)
    fb_answer, fb_lat = _nl_fallback(client, prompt, category, model)
    return fb_answer, "math_nl_fallback", model, fb_lat


# ---------------------------------------------------------------------------
# Logic: direct Fireworks NL (no code-exec)
# ---------------------------------------------------------------------------

def _solve_logic_nl(client, prompt: str):
    """Send logic puzzles directly to the reasoning specialist as NL."""
    model    = route("logic_puzzle")
    messages = normalize_prompt(prompt, "logic_puzzle")
    messages = _maybe_compress(messages, prompt)
    max_tok  = get_max_tokens("logic_puzzle")
    answer, lat = client.call(model, messages, max_tokens=max_tok,
                              timeout=PER_REQUEST_TIMEOUT_S)
    answer = answer or ""
    if not answer and not lat.get("error"):
        # Model returned empty — try once more with a more explicit instruction.
        messages[0]["content"] = (
            "You are a logical reasoning expert. Work through the puzzle step by step, "
            "stating which option each clue eliminates. State the final answer on the last line."
        )
        answer, lat = client.call(model, messages, max_tokens=max_tok,
                                  timeout=PER_REQUEST_TIMEOUT_S)
        answer = answer or ""
    return answer, "logic_nl", model, lat


# ---------------------------------------------------------------------------
# Code generation / debugging with verification
# ---------------------------------------------------------------------------

def _solve_via_code_verify(client, prompt: str, category: str):
    """Generate code → verify (AST + exec) → auto-fix → NL fallback."""
    model    = route(category)
    messages = normalize_prompt(prompt, category)
    messages = _maybe_compress(messages, prompt)
    max_tok  = get_max_tokens(category)
    llm_out, lat = client.call(model, messages, max_tokens=max_tok,
                               timeout=PER_REQUEST_TIMEOUT_S)

    if lat.get("error") or not llm_out:
        fb_answer, fb_lat = _nl_fallback(client, prompt, category, model)
        return fb_answer, "code_verify_fallback", model, fb_lat

    # Guard against truncated output that has no code block at all.
    if "```" not in llm_out:
        print(f"[code] output has no code block, retrying with stricter prompt",
              file=sys.stderr)
        messages[0]["content"] = (
            "Return ONLY a Python code block starting with ```python and ending with ```. "
            "No text before or after. Implement: " + prompt[:200]
        )
        llm_out, lat2 = client.call(model, messages, max_tokens=max_tok,
                                    timeout=PER_REQUEST_TIMEOUT_S)
        lat.update(lat2)
        if not llm_out or "```" not in llm_out:
            fb_answer, fb_lat = _nl_fallback(client, prompt, category, model)
            return fb_answer, "code_verify_fallback", model, fb_lat

    final_answer, path, verify_lat = verify_and_fix(
        llm_output=llm_out, prompt=prompt, category=category,
        client=client, model=model, max_retries=2,
        timeout=PER_REQUEST_TIMEOUT_S,
    )
    lat.update(verify_lat)
    return final_answer or "", path, model, lat


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _nl_fallback(client, prompt: str, category: str, model: str):
    """Plain NL Fireworks call — last resort when all other paths fail."""
    messages = normalize_prompt(prompt, category)
    messages = _maybe_compress(messages, prompt)
    max_tok  = get_max_tokens(category)
    answer, lat = client.call(model, messages, max_tokens=max_tok,
                              timeout=PER_REQUEST_TIMEOUT_S)
    return answer or "", lat


def _maybe_compress(messages: list, prompt: str) -> list:
    """
    Compress the user message only when it's genuinely long.
    Logs a note when compression fires so it's visible in stderr.
    """
    word_count = len(prompt.split())
    if word_count <= COMPRESS_THRESHOLD:
        return messages
    before = estimate_tokens(prompt)
    compressed = compress_messages(messages, max_tokens=COMPRESS_THRESHOLD)
    after_content = next(
        (m["content"] for m in compressed if m["role"] == "user"), prompt
    )
    after = estimate_tokens(after_content)
    print(f"[compress] {before} → {after} est. tokens", file=sys.stderr)
    return compressed


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
