"""
main.py -- container entrypoint.

Routing logic:
    sentiment            → local Qwen2.5-1.5B (zero Fireworks tokens)
    factual knowledge    → local Qwen2.5-1.5B (zero Fireworks tokens)
    summarization        → Fireworks gemma-4-26b-a4b-it (tier-0), escalates to
                           gemma-4-31b-it (tier-1) then minimax-m3 (tier-2) on poor answer
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
    route, route_summarization_next,
    CODE_EXEC_CATEGORIES, LOGIC_NL_CATEGORIES,
    get_allowed_models, resolve_roles,
)
from categories.code_exec import (
    build_codegen_messages, extract_code, run_code_safely,
)
from categories.code_verifier import verify_and_fix
from categories.prompt_compressor import compress_messages, estimate_tokens
from categories.executor import get_task_executor, get_max_workers, ENV_PROFILE
from categories.metrics import MetricsCollector
from categories.local_model import get_local_model, run_sentiment, run_factual
from categories.code_exec import (
    build_codegen_messages, extract_code, run_code_safely, CODE_GEN_SYSTEM,
)
from categories.local_model import get_local_model, run_sentiment, run_factual, run_summarization, run_logic

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
    # Explicit refusals / apologies are always bad.
    if any(marker in lowered for marker in ("i cannot", "i can't", "sorry", "unable", "refuse")):
        return True
    # Bullet-point and multi-line summaries are VALID — don't penalise newlines.
    # Only flag if it looks like uncontrolled verbosity: no bullets but very long prose.
    words = answer.split()
    bullet_lines = sum(1 for line in answer.splitlines() if line.strip().startswith(("-", "•", "*", "·")))
    is_bullet_format = bullet_lines >= 2
    if not is_bullet_format and len(words) > 150:
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
    """
    The model now returns clean markdown (## Bug Explanation / ## Corrected Code).
    Strip any leftover JSON wrapper if it accidentally appears, then return
    the answer as-is so the LLM judge sees human-readable markdown.
    """
    # If the model still returned a JSON blob despite the new instruction,
    # parse it and render it as markdown so we never return raw JSON.
    candidate = _extract_json_candidate(answer, "{", "}")
    if candidate:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict) and ("issues" in parsed or "corrected_parts" in parsed):
            issues = parsed.get("issues", [])
            if isinstance(issues, str):
                issues = [issues]
            if not isinstance(issues, list):
                issues = []

            corrected_parts = parsed.get("corrected_parts", [])
            code_snippets = []
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
                            code_snippets.append(snippet.strip())
                    elif isinstance(part, str) and part.strip():
                        code_snippets.append(part.strip())

            if not code_snippets:
                raw = extract_code(answer)
                if raw.strip():
                    code_snippets.append(raw.strip())

            issue_text = "\n".join(f"- {i}" for i in issues) if issues else "- The code contained a bug."
            code_text  = "\n\n".join(f"```python\n{s}\n```" for s in code_snippets) if code_snippets else ""
            return f"## Bug Explanation\n{issue_text}\n\n## Corrected Code\n{code_text}".strip()

    # Answer is already markdown (or plain text) — return directly.
    if not (answer or "").strip():
        return _last_resort_answer("code_debugging", prompt)
    return answer.strip()


# ---------------------------------------------------------------------------
# Math: detect if the prompt asks for reasoning/steps
# ---------------------------------------------------------------------------
_REASONING_KEYWORDS = re.compile(
    r"\b(show|explain|step(?:s)?|reasoning|walk(?:[-\s]?through)?|how|why|work(?:[-\s]?out)?)"
    r"|step[-\s]by[-\s]step",
    re.IGNORECASE,
)


def _prompt_wants_steps(prompt: str) -> bool:
    """Return True if the prompt explicitly asks for steps or explanation."""
    return bool(_REASONING_KEYWORDS.search(prompt))



# ---------------------------------------------------------------------------
# Per-task processing
# ---------------------------------------------------------------------------

def process_task(client, task: dict, metrics: MetricsCollector, local_model) -> dict:
    task_id  = task["task_id"]
    prompt   = task["prompt"]
    category = classify(prompt, local_model)

    # ── Local model path (sentiment / factual only — summarization is Fireworks-only) ──
    if category in LOCAL_CATEGORIES and local_model is not None and local_model.is_loaded():
        t0 = time.time()
        try:
            if category == "sentiment":
                answer = run_sentiment(prompt, local_model)
            elif category == "factual_knowledge":
                answer = run_factual(prompt, local_model)
            elif category == "summarization":
                answer = run_summarization(prompt, local_model)
            elif category == "logic_puzzle":
                answer = run_logic(prompt, local_model)
            else:
                answer = ""
        except Exception as exc:
            print(f"[local] {task_id} {category} failed: {exc}", file=sys.stderr)
            answer = ""
        elapsed_ms = round((time.time() - t0) * 1000, 1)

        if answer and answer.strip():
            lat = {
                "total_latency_ms": elapsed_ms,
                "ttft_ms": None, "prompt_tokens": None,
                "completion_tokens": None, "total_tokens": None, "error": None,
            }
            answer = _ensure_nonempty_answer(answer, category, prompt)
            metrics.log(task_id, category, f"local_{category}", "local_qwen2.5", lat)
            return {"task_id": task_id, "answer": answer}
        else:
            print(f"[local] {task_id} {category}: empty or failed, "
                  f"falling back to Fireworks", file=sys.stderr)
            # Fall through to the Fireworks paths below

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

    # ── Summarization: Fireworks-only, gemma primary -> minimax direct fallback ──
    # No local-model attempt, no cascade through other gemma variants, and no
    # canned filler text if both calls come back empty — we just return
    # whatever the best real attempt produced.
    if category == "summarization":
        model    = route("summarization")   # resolves to gemma-4-26b-a4b-it when healthy
        messages = normalize_prompt(prompt, "summarization")
        messages = _maybe_compress(messages, prompt)
        max_tok  = get_max_tokens("summarization")
        answer, lat = client.call(model, messages, max_tokens=max_tok,
                                  timeout=PER_REQUEST_TIMEOUT_S)
        answer = answer or ""

        # Escalate to another real model call whenever the current answer
        # looks empty/off/refused — but always escalate to a live generation,
        # never to a canned string. A low-quality real answer beats a
        # placeholder for the judge.
        if not answer.strip() or lat.get("error") or _summarization_seems_off(answer):
            fallback_model = route_summarization_next(model)
            if fallback_model and fallback_model != model:
                print(f"[summarization] {task_id}: {model} failed/poor, "
                      f"falling back directly to {fallback_model}", file=sys.stderr)
                fb_answer, fb_lat = client.call(fallback_model, messages, max_tokens=max_tok,
                                                timeout=PER_REQUEST_TIMEOUT_S)
                fb_answer = fb_answer or ""
                if fb_answer.strip():
                    answer, lat, model = fb_answer, fb_lat, fallback_model

        # Cascade still empty — force ONE more real generation with an
        # explicit "you must answer" instruction rather than giving up.
        # Cascade still empty — force ONE more real generation, but make it
        # structurally different from the failed attempts: give it a much
        # bigger token budget (the prior failures may have been a reasoning
        # model running out of budget mid-thought, not a real refusal) and
        # explicitly forbid visible reasoning so the budget goes to the answer.
        if not answer.strip():
            print(f"[summarization] {task_id}: cascade still empty "
                  f"(last error: {lat.get('error')}), forcing one more attempt "
                  f"with expanded budget", file=sys.stderr)
            recovery_messages = [
                {"role": "system", "content": (
                    "Summarize the text in 1-3 sentences. Respond with ONLY the "
                    "summary text -- no reasoning, no preamble, no explanation "
                    "of your approach."
                )},
                {"role": "user", "content": prompt},
            ]
            answer, lat = client.call(model, recovery_messages,
                                      max_tokens=max(max_tok * 2, 500),
                                      timeout=PER_REQUEST_TIMEOUT_S)
            answer = answer or ""

        metrics.log(task_id, category, "fireworks_summarization", model, lat)
        return {"task_id": task_id, "answer": answer}

        metrics.log(task_id, category, "fireworks_summarization", model, lat)
        return {"task_id": task_id, "answer": answer}
    # ── General Fireworks path (factual, NER fallback, …) ───────────────
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
    Math reasoning stays almost entirely on the local-execution path:

      1. Ask the code specialist for a short Python script (one cheap call).
      2. Run it locally — zero extra tokens, deterministic result.
      3. If it errors, ask for ONE fix retry using the traceback (same idea
         as the code_generation verifier) before giving up on code entirely.
      4. Only if both code attempts fail do we fall back to NL — and even
         then we only ask for step-by-step reasoning if the prompt actually
         requested it. Otherwise we ask for just the final value.
    """
    model    = route(category)
    messages = build_codegen_messages(prompt)
    llm_out, lat = client.call(model, messages, max_tokens=400,
                               timeout=PER_REQUEST_TIMEOUT_S)

    code = extract_code(llm_out) if (llm_out and not lat.get("error")) else ""
    ok, output = (False, "")
    if code:
        ok, output = run_code_safely(code, timeout_s=CODE_EXEC_TIMEOUT_S)

    # One fix-style retry using the actual error, instead of jumping
    # straight to a full NL answer.
    if not (ok and output) and code:
        print(f"[math] first script failed ({output[:80]!r}), retrying with fix prompt",
              file=sys.stderr)
        fix_messages = [
            {"role": "system", "content": CODE_GEN_SYSTEM},
            {"role": "user", "content": (
                f"{prompt}\n\nYour previous script failed with this error:\n{output}\n\n"
                "Fix the script and return only the corrected ```python block."
            )},
        ]
        fix_out, fix_lat = client.call(model, fix_messages, max_tokens=400,
                                        timeout=PER_REQUEST_TIMEOUT_S)
        if fix_out and not fix_lat.get("error"):
            fix_code = extract_code(fix_out)
            if fix_code:
                ok, output = run_code_safely(fix_code, timeout_s=CODE_EXEC_TIMEOUT_S)
        lat.update({f"fix_{k}": v for k, v in fix_lat.items() if k not in lat})

    if ok and output:
        print(f"[math] local exec succeeded: {output[:60]}", file=sys.stderr)
        if _prompt_wants_steps(prompt):
            print(f"[math] prompt requests steps — fetching NL explanation", file=sys.stderr)
            exp_messages = normalize_prompt(prompt, "math_reasoning")
            exp_messages = _maybe_compress(exp_messages, prompt)
            exp_out, _ = client.call(model, exp_messages, max_tokens=get_max_tokens("math_reasoning"),
                                     timeout=PER_REQUEST_TIMEOUT_S)
            if exp_out and exp_out.strip():
                exp_clean = re.sub(r"\n*Answer:.*\Z", "", exp_out.strip(),
                                   flags=re.IGNORECASE | re.DOTALL).strip()
                combined = f"{exp_clean}\n\nAnswer: {output}"
                return combined, "math_local_exec_with_explanation", model, lat
        return output, "math_local_exec", model, lat

    # Both code attempts failed — last resort is NL, kept as terse as the
    # code path unless the prompt explicitly asked for steps.
    print(f"[math] local exec failed twice, falling back to NL", file=sys.stderr)
    if _prompt_wants_steps(prompt):
        fb_answer, fb_lat = _nl_fallback(client, prompt, category, model)
    else:
        terse_messages = [
            {"role": "system", "content": "Solve the problem and reply with ONLY the final numeric value, nothing else."},
            {"role": "user", "content": prompt},
        ]
        fb_answer, fb_lat = client.call(model, terse_messages, max_tokens=60,
                                        timeout=PER_REQUEST_TIMEOUT_S)
        fb_answer = fb_answer or ""
    return fb_answer, "math_nl_fallback", model, fb_lat

def _solve_logic_nl(client, prompt: str):
    model    = route("logic_puzzle")
    max_tok  = get_max_tokens("logic_puzzle")
    messages = normalize_prompt(prompt, "logic_puzzle")
    messages = _maybe_compress(messages, prompt)

    answer, lat = client.call(model, messages, max_tokens=max_tok,
                              timeout=PER_REQUEST_TIMEOUT_S)
    answer = answer or ""

    # If minimax exhausted its budget on reasoning without producing content,
    # retry with a stripped-down prompt that gives it no room to elaborate.
    if not answer.strip() or lat.get("error"):
        print(f"[logic] budget exhausted or empty, retrying with terse prompt", file=sys.stderr)
        terse_messages = [
            {"role": "system", "content": (
                "Answer in 3 sentences maximum. "
                "State only the final answer. No reasoning trace, no bullet points, no code. "
                "End with 'Answer:' followed by the solution."
            )},
            {"role": "user", "content": prompt},
        ]
        answer, lat = client.call(model, terse_messages, max_tokens=400,
                                  timeout=PER_REQUEST_TIMEOUT_S)
        answer = answer or ""

    if answer.strip() and "answer:" not in answer.lower():
        strict_messages = [
            {"role": "system", "content": (
                "State ONLY the final answer to this logic puzzle in one or two "
                "plain-language sentences. No code, no reasoning trace, no clue restatement."
            )},
            {"role": "user", "content": prompt},
        ]
        retry_answer, retry_lat = client.call(model, strict_messages, max_tokens=max_tok,
                                              timeout=PER_REQUEST_TIMEOUT_S)
        if retry_answer and retry_answer.strip():
            answer, lat = retry_answer, retry_lat

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

    # Detect reasoning-model truncation: code block started but got cut off
    if llm_out and "```" in llm_out:
        extracted = extract_code(llm_out)
        try:
            import ast as _ast
            _ast.parse(extracted)
            syntax_ok = True
        except SyntaxError:
            syntax_ok = False

        if not syntax_ok:
            print(f"[code] syntax check failed — likely truncation, retrying with 2× token budget",
              file=sys.stderr)
        expanded_tok = min(max_tok * 2, 2048)
        llm_out2, lat2 = client.call(model, messages, max_tokens=expanded_tok,
                                     timeout=PER_REQUEST_TIMEOUT_S)
        if llm_out2 and "```" in llm_out2:
            llm_out = llm_out2
            lat.update(lat2)

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
        "sentiment": " Reply with one sentence: start with the label (positive, negative, or mixed) then briefly justify it.",
        "summarization": " Return a non-empty summary that obeys the requested format.",
        "ner": " Return a valid JSON array. If there are no entities, return [].",
        "code_debugging": " Return a markdown explanation with a Bug Explanation section and a Corrected Code section.",
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