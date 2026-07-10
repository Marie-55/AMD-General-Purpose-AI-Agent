"""
main.py -- container entrypoint.

Routing logic:
    sentiment            → local Qwen2.5-1.5B (zero Fireworks tokens)
    factual knowledge    → local Qwen2.5-1.5B (zero Fireworks tokens)
    summarization        → local Qwen2.5-1.5B, Fireworks fallback if off
    ner                  → Fireworks cheap_general
    math_reasoning       → local Python execution first (zero tokens);
                                                 Fireworks NL fallback only if local exec fails
    logic_puzzle         → Fireworks reasoning specialist
    code_gen / debug     → Fireworks code specialist + AST/exec verification

max_tokens is dynamic per category (see normalizer.CATEGORY_MAX_TOKENS).
Summarization is shaped in the prompt and accepted directly; no exact word-count
post-validation is performed.
"""
import json
import os
import re
import sys
import time
from concurrent.futures import as_completed

from categories.classifier import classify
from categories.normalizer import normalize_prompt, get_max_tokens
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
from categories.local_model import get_local_model, run_sentiment, run_factual, run_summarization

# Hard limits from the competition rules.
TOTAL_TIME_BUDGET_S   = 9 * 60
PER_REQUEST_TIMEOUT_S = 25
CODE_EXEC_TIMEOUT_S   = 8.0
MAX_WORKERS           = get_max_workers()

# Categories answered entirely by the local Qwen model — zero Fireworks tokens.
LOCAL_CATEGORIES = {"sentiment", "factual_knowledge", "summarization"}

# Categories routed through code-gen + AST/exec verification.
CODE_VERIFY_CATEGORIES = {"code_generation", "code_debugging"}

# Prompt compression threshold: only compress user messages longer than this.
COMPRESS_THRESHOLD = 300   # words


def _last_resort_answer(category: str, prompt: str) -> str:
    """Return a category-specific non-empty fallback if all other paths fail."""
    if category == "sentiment":
        return "neutral"
    if category == "ner":
        return "[]"
    if category == "summarization":
        return "I could not determine a reliable answer."
    if category == "math_reasoning":
        return "0"
    if category == "code_debugging":
        return json.dumps(
            {
                "issues": ["The code needs a correction, but I could not verify a safe fix."],
                "corrected_parts": [],
            },
            ensure_ascii=False,
        )
    if category == "code_generation":
        return "```python\npass\n```"
    if category == "logic_puzzle":
        return "I could not determine a unique solution."
    return "I could not determine a reliable answer."


def _ensure_nonempty_answer(answer: str, category: str, prompt: str) -> str:
    return answer if (answer or "").strip() else _last_resort_answer(category, prompt)


def _summarization_seems_off(answer: str) -> bool:
    if not answer or not answer.strip():
        return True
    lowered = answer.lower()
    if any(marker in lowered for marker in ("i cannot", "i can't", "sorry", "unable", "refuse")):
        return True
    if answer.count("\n") > 3:
        return True
    if len(answer.split()) > 120:
        return True
    return False


def _extract_json_candidate(text: str, opening: str, closing: str) -> str:
    if not text:
        return ""

    fenced = re.search(rf"```json\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced.group(1).strip()

    start = text.find(opening)
    if start == -1:
        return ""
    candidate = text[start:].strip()
    end = candidate.rfind(closing)
    if end != -1:
        candidate = candidate[: end + 1]
    return candidate.strip()


def _normalize_ner_answer(answer: str, prompt: str) -> str:
    candidate = _extract_json_candidate(answer, "[", "]")
    if candidate:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            normalized = []
            seen = set()
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                text_value = str(item.get("text", "")).strip()
                type_value = str(item.get("type", "")).upper().strip()
                if not text_value or type_value not in {"PERSON", "ORG", "LOCATION", "DATE"}:
                    continue
                key = (text_value, type_value)
                if key in seen:
                    continue
                seen.add(key)
                normalized.append({"text": text_value, "type": type_value})
            return json.dumps(normalized, ensure_ascii=False)
    return "[]"


def _normalize_code_debug_answer(answer: str, prompt: str) -> str:
    candidate = _extract_json_candidate(answer, "{", "}")
    if candidate:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            issues = parsed.get("issues", [])
            if isinstance(issues, str):
                issues = [issues]
            if not isinstance(issues, list):
                issues = []

            corrected_parts = parsed.get("corrected_parts", [])
            normalized_parts = []
            if isinstance(corrected_parts, list):
                for part in corrected_parts:
                    if isinstance(part, dict):
                        snippet = (
                            part.get("corrected_code")
                            or part.get("code")
                            or part.get("fix")
                            or part.get("corrected")
                        )
                        if isinstance(snippet, str) and snippet.strip():
                            item = {"corrected_code": snippet.strip()}
                            location = part.get("location")
                            original = part.get("original")
                            if isinstance(location, str) and location.strip():
                                item["location"] = location.strip()
                            if isinstance(original, str) and original.strip():
                                item["original"] = original.strip()
                            normalized_parts.append(item)
                    elif isinstance(part, str) and part.strip():
                        normalized_parts.append({"corrected_code": part.strip()})

            if not normalized_parts:
                extracted_code = extract_code(answer)
                if extracted_code.strip():
                    normalized_parts = [{"corrected_code": extracted_code.strip()}]

            if not issues:
                issues = ["The original code had a bug."]

            normalized = {
                "issues": [str(item).strip() for item in issues if str(item).strip()],
                "corrected_parts": normalized_parts,
            }
            return json.dumps(normalized, ensure_ascii=False)

    extracted_code = extract_code(answer)
    if not extracted_code.strip():
        extracted_code = _last_resort_answer("code_debugging", prompt)
    return json.dumps(
        {
            "issues": ["The original code had a bug."],
            "corrected_parts": [{"corrected_code": extracted_code.strip()}],
        },
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# Per-task processing
# ---------------------------------------------------------------------------

def process_task(client, task: dict, metrics: MetricsCollector, local_model) -> dict:
    task_id  = task["task_id"]
    prompt   = task["prompt"]
    category = classify(prompt, local_model)

    # ── Local model path (sentiment / factual / summarization) ────────────
    if category in LOCAL_CATEGORIES and local_model is not None and local_model.is_loaded():
        t0 = time.time()
        try:
            if category == "sentiment":
                answer = run_sentiment(prompt, local_model)
            elif category == "factual_knowledge":
                answer = run_factual(prompt, local_model)
            else:
                answer = run_summarization(prompt, local_model)
        except Exception as exc:
            print(f"[local] {task_id} {category} failed: {exc}", file=sys.stderr)
            answer = ""
        elapsed_ms = round((time.time() - t0) * 1000, 1)

        if answer and not (category == "summarization" and _summarization_seems_off(answer)):
            if category == "ner":
                answer = _normalize_ner_answer(answer, prompt)
            lat = {
                "total_latency_ms": elapsed_ms,
                "ttft_ms": None, "prompt_tokens": None,
                "completion_tokens": None, "total_tokens": None, "error": None,
            }
            answer = _ensure_nonempty_answer(answer, category, prompt)
            metrics.log(task_id, category, f"local_{category}", "local_qwen2.5", lat)
            return {"task_id": task_id, "answer": answer}
        else:
            print(f"[local] {task_id} {category}: empty or off, falling back to Fireworks",
                  file=sys.stderr)

    # ── Math: local Python execution first (zero Fireworks tokens) ───────
    if category in CODE_EXEC_CATEGORIES:           # {"math_reasoning"}
        answer, path, model, lat = _solve_math(client, prompt, category)
        if not (answer or "").strip():
            answer, lat = _recover_empty_answer(client, prompt, category, model)
        answer = _ensure_nonempty_answer(answer, category, prompt)
        metrics.log(task_id, category, path, model, lat)
        return {"task_id": task_id, "answer": answer}

    # ── Logic: Fireworks NL reasoning ─────────────────────────────────────
    if category in LOGIC_NL_CATEGORIES:
        answer, path, model, lat = _solve_logic_nl(client, prompt)
        if not (answer or "").strip():
            answer, lat = _recover_empty_answer(client, prompt, category, model)
        answer = _ensure_nonempty_answer(answer, category, prompt)
        metrics.log(task_id, category, path, model, lat)
        return {"task_id": task_id, "answer": answer}

    # ── Code generation / debugging with verification ───────────────────────
    if category in CODE_VERIFY_CATEGORIES:
        answer, path, model, lat = _solve_via_code_verify(client, prompt, category)
        if category == "code_debugging":
            answer = _normalize_code_debug_answer(answer, prompt)
        if not (answer or "").strip():
            answer, lat = _recover_empty_answer(client, prompt, category, model)
        answer = _ensure_nonempty_answer(answer, category, prompt)
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

    if category == "ner":
        answer = _normalize_ner_answer(answer, prompt)

    if not answer.strip():
        answer, lat = _recover_empty_answer(client, prompt, category, model)
        answer = answer or ""

    if category == "ner":
        answer = _normalize_ner_answer(answer, prompt)

    if lat.get("error"):
        answer = answer or ""
    answer = _ensure_nonempty_answer(answer, category, prompt)
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


def _recover_empty_answer(client, prompt: str, category: str, model: str):
    """Retry once with a stricter prompt when a task returns an empty answer."""
    messages = normalize_prompt(prompt, category)
    messages = _maybe_compress(messages, prompt)

    extra = {
        "factual_knowledge": " Answer directly and do not return an empty response.",
        "math_reasoning": " Provide the final numeric answer even if the reasoning is brief.",
        "sentiment": " Return exactly one word: positive, negative, or mixed.",
        "summarization": " Return a non-empty summary that obeys the requested format.",
        "ner": " Return a valid JSON array. If there are no entities, return [].",
        "code_debugging": " Return valid JSON with issues and corrected_parts.",
        "logic_puzzle": " State one valid final answer clearly.",
        "code_generation": " Return a complete Python code block only.",
    }
    messages[0]["content"] = messages[0]["content"] + extra.get(
        category,
        " Return a non-empty answer.",
    )

    max_tok = get_max_tokens(category)
    answer, lat = client.call(model, messages, max_tokens=max_tok,
                              timeout=PER_REQUEST_TIMEOUT_S)
    return answer or "", lat


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
