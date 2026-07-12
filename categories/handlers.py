"""One function per category, each taking (task, runtime) and returning an
answer string. All share the ModelRuntime instance main.py loads once per
model bucket.
"""
import ast
import json
import re

import config
from categories import prompts, grammars, math_solver
from categories.code_exec import extract_code
from categories.task_categories import (
    FACTUAL_KNOWLEDGE,
    MATH_REASONING,
    SENTIMENT,
    SUMMARIZATION,
    NER,
    CODE_DEBUGGING,
    LOGIC_PUZZLE,
    CODE_GENERATION,
)


def _looks_incomplete(text: str) -> bool:
    text = (text or "").strip()
    if not text:
        return True
    return text[-1] not in ".!?\"')]}" and not text.endswith("```")


def handle_factual(task, runtime):
    messages = [
        {"role": "system", "content": prompts.FACTUAL_SYSTEM},
        {"role": "user", "content": task["prompt"]},
    ]
    answer = runtime.generate(messages, max_tokens=config.FACTUAL_MAX_TOKENS, temperature=0.3)
    return answer or config.FALLBACK_ANSWER


def handle_sentiment(task, runtime):
    messages = [
        {"role": "system", "content": prompts.SENTIMENT_SYSTEM},
        {"role": "user", "content": task["prompt"]},
    ]
    answer = runtime.generate(messages, max_tokens=config.SENTIMENT_MAX_TOKENS, temperature=0.3)
    if not answer:
        return config.FALLBACK_ANSWER

    # Phi-4-mini can produce verbose output -- ensure a valid label is present.
    # If the raw answer already starts with a known label, return as-is.
    label_pat = re.compile(r"\b(positive|negative|neutral|mixed)\b", re.IGNORECASE)
    if not label_pat.search(answer[:80]):
        # Model may have buried the label; try to surface it.
        m = label_pat.search(answer)
        if m:
            label = m.group(1).capitalize()
            # Rebuild: label + the first sentence after it (for the justification).
            rest = answer[m.end():].strip().lstrip(".:- \t")
            # Grab up to first sentence boundary.
            sent_end = re.search(r"[.!?]", rest)
            justification = rest[: sent_end.end()] if sent_end else rest[:120]
            answer = f"{label}. {justification}" if justification else label

    return answer


def handle_summarization(task, runtime):
    prompt = task["prompt"]
    messages = [
        {"role": "system", "content": prompts.SUMMARIZATION_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    answer = runtime.generate(messages, max_tokens=config.SUMMARIZATION_MAX_TOKENS, temperature=0.3)

    # Never trim to enforce a constraint -- only re-prompt if the first
    # attempt looks empty or cut off mid-sentence.
    if _looks_incomplete(answer):
        retry_messages = [
            {"role": "system", "content": prompts.SUMMARIZATION_RETRY_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        retry = runtime.generate(
            retry_messages, max_tokens=config.SUMMARIZATION_RETRY_MAX_TOKENS, temperature=0.3
        )
        if retry and retry.strip():
            answer = retry

    return (answer or "").strip() or config.FALLBACK_ANSWER


def _repair_ner_json(raw: str) -> str:
    """Try to extract a JSON array from noisy model output."""
    # 1. Try direct parse.
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return raw
    except (json.JSONDecodeError, TypeError):
        pass

    # 2. Try to slice out just the [...] portion.
    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidate = raw[start: end + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                return candidate
        except (json.JSONDecodeError, TypeError):
            pass

    # 3. Extract entity/type pairs manually with regex as last resort.
    entities = []
    for m in re.finditer(
        r'"text"\s*:\s*"([^"]+)"\s*,\s*"type"\s*:\s*"([^"]+)"', raw
    ):
        entities.append({"text": m.group(1), "type": m.group(2)})
    if entities:
        return json.dumps(entities)

    return "[]"


def handle_ner(task, runtime):
    messages = [
        {"role": "system", "content": prompts.NER_SYSTEM},
        {"role": "user", "content": task["prompt"]},
    ]
    raw = runtime.generate(
        messages,
        max_tokens=config.NER_MAX_TOKENS,
        temperature=0.1,
        grammar=grammars.get_ner_grammar(),
    )

    try:
        json.loads(raw)
        return raw
    except (json.JSONDecodeError, TypeError):
        # Grammar should have prevented this, but if the model still
        # produces garbage (e.g. empty string on OOM), repair it.
        return _repair_ner_json(raw or "")


def handle_logic(task, runtime):
    messages = [
        {"role": "system", "content": prompts.LOGIC_SYSTEM},
        {"role": "user", "content": task["prompt"]},
    ]
    answer = runtime.generate(messages, max_tokens=config.LOGIC_MAX_TOKENS, temperature=0.2)
    if not answer:
        return config.FALLBACK_ANSWER

    # If model didn't produce an "Answer:" line, retry with a terse prompt.
    if "answer:" not in answer.lower():
        retry_messages = [
            {
                "role": "system",
                "content": (
                    "Solve the puzzle. State your conclusion on the last line as: "
                    "Answer: <answer>. Be concise."
                ),
            },
            {"role": "user", "content": task["prompt"]},
        ]
        retry = runtime.generate(retry_messages, max_tokens=400, temperature=0.1)
        if retry and retry.strip():
            answer = retry

    return answer or config.FALLBACK_ANSWER


def handle_math(task, runtime):
    return math_solver.solve(task["prompt"], runtime)


def handle_code_debugging(task, runtime):
    messages = [
        {"role": "system", "content": prompts.CODE_DEBUG_SYSTEM},
        {"role": "user", "content": task["prompt"]},
    ]
    answer = runtime.generate(messages, max_tokens=config.CODE_DEBUG_MAX_TOKENS, temperature=0.2)
    return answer or config.FALLBACK_ANSWER


def handle_code_generation(task, runtime):
    prompt = task["prompt"]
    messages = [
        {"role": "system", "content": prompts.CODE_GEN_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    answer = runtime.generate(messages, max_tokens=config.CODE_GEN_MAX_TOKENS, temperature=0.2)

    code = extract_code(answer)
    if code:
        try:
            ast.parse(code)
        except SyntaxError as exc:
            # Syntax error OR truncation -- retry with double budget.
            fix_messages = messages + [
                {"role": "assistant", "content": answer},
                {
                    "role": "user",
                    "content": (
                        f"That code has a syntax error ({exc}). "
                        "Provide the complete, corrected code in a single Python code block."
                    ),
                },
            ]
            fixed = runtime.generate(
                fix_messages,
                max_tokens=min(config.CODE_GEN_MAX_TOKENS * 2, 2000),
                temperature=0.2,
            )
            if fixed and fixed.strip():
                answer = fixed
    elif not answer:
        # No code block and no answer -- direct retry.
        retry_messages = [
            {"role": "system", "content": prompts.CODE_GEN_SYSTEM},
            {
                "role": "user",
                "content": prompt + "\n\nReturn ONLY a Python code block. No prose.",
            },
        ]
        retry = runtime.generate(
            retry_messages, max_tokens=config.CODE_GEN_MAX_TOKENS, temperature=0.2
        )
        if retry and retry.strip():
            answer = retry

    return (answer or "").strip() or config.FALLBACK_ANSWER


DISPATCH = {
    FACTUAL_KNOWLEDGE: handle_factual,
    SENTIMENT: handle_sentiment,
    SUMMARIZATION: handle_summarization,
    NER: handle_ner,
    LOGIC_PUZZLE: handle_logic,
    MATH_REASONING: handle_math,
    CODE_DEBUGGING: handle_code_debugging,
    CODE_GENERATION: handle_code_generation,
}
