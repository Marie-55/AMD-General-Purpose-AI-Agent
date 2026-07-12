"""Logic puzzles: local model first, escalate to Fireworks NL reasoning
(with a truncation-rescue pass) if the local attempt isn't confident."""
import re
import sys

from categories.local_model import run_logic
from categories.normalizer import normalize_prompt, get_max_tokens
from categories.routing import route
from categories import prompts
from config import PER_REQUEST_TIMEOUT_S, LOGIC_TIMEOUT_S
from categories.handlers.base import (
    try_local_first,
    ensure_nonempty_answer,
    maybe_compress,
    recover_empty_answer,
    VAGUE_MARKERS,
)

_MIN_CONFIDENT_WORDS = 15


def _logic_answer_is_confident(answer: str) -> bool:
    """
    Mirrors the factual_knowledge confidence gate: accept a local logic
    answer only if it looks like a real, concluded solution rather than a
    refusal or a reasoning trace that never reached a final answer.
    """
    if not (answer or "").strip():
        return False
    if "answer:" not in answer.lower():
        return False
    if len(answer.split()) < _MIN_CONFIDENT_WORDS:
        return False
    lowered = answer.lower()
    if any(m in lowered for m in VAGUE_MARKERS):
        return False
    return True


def _rescue_truncated_logic(answer: str) -> str:
    """
    If the response was cut off before 'Answer:', extract the last complete
    sentence that looks like a conclusion and append it as the answer line.
    Handles the t19 pattern: reasoning stops mid-sentence with no Answer: line.
    """
    if not answer:
        return answer
    if re.search(r"\bAnswer\s*:", answer, re.IGNORECASE):
        return answer  # already has an answer line — nothing to rescue

    # Split into sentences (rough but sufficient)
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    complete = [s.strip() for s in sentences if s.strip() and s.strip()[-1] in ".!?"]
    if not complete:
        return answer

    # Walk back from the end to find the last sentence that contains a
    # concrete noun/assignment (person, color, position keywords)
    conclusion_re = re.compile(
        r"\b(must|therefore|so|thus|hence|drinks?|owns?|sits?|is in|lives? in"
        r"|comes? in|finishes?|placed|assigned|from left|from right"
        r"|order is|position is|answer is)\b",
        re.IGNORECASE,
    )
    for sent in reversed(complete):
        if conclusion_re.search(sent):
            return answer.rstrip() + f"\n\nAnswer: {sent}"

    # Fall back: just append the last complete sentence
    return answer.rstrip() + f"\n\nAnswer: {complete[-1]}"


def _solve_logic_nl(client, prompt: str, category: str):
    """
    Single Fireworks call for logic puzzles.
    Uses LOGIC_TIMEOUT_S (50s) — the previous 25s limit caused t19 truncation.
    Rescues truncated responses that have no Answer: line via _rescue_truncated_logic.
    """
    model = route(category)
    max_tok = get_max_tokens(category)   # 4096 — let it reason fully
    messages = normalize_prompt(prompt, category)
    messages = maybe_compress(messages, prompt)

    answer, lat = client.call(model, messages, max_tokens=max_tok,
                              timeout=LOGIC_TIMEOUT_S)
    answer = (answer or "").strip()

    # Hard failure: empty or API error → one terse retry
    if not answer or lat.get("error"):
        print(f"[logic] empty/error on first call, one terse retry", file=sys.stderr)
        terse_messages = [
            {"role": "system", "content": prompts.LOGIC_TERSE_RETRY_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        answer, lat = client.call(model, terse_messages, max_tokens=max_tok,
                                  timeout=LOGIC_TIMEOUT_S)
        answer = (answer or "").strip()

    # Rescue truncated responses that are missing the Answer: line
    answer = _rescue_truncated_logic(answer)

    return answer, "logic_nl", model, lat


def handle(task: dict, category: str, ctx) -> dict:
    task_id = task["task_id"]
    prompt = task["prompt"]

    local_answer = try_local_first(
        ctx, task_id, category, prompt,
        local_fn=run_logic,
        is_good_enough=_logic_answer_is_confident,
        path_name="local_logic",
    )
    if local_answer is not None:
        return {"task_id": task_id, "answer": local_answer}

    print(f"[logic] {task_id}: local empty/low-confidence, escalating to Fireworks", file=sys.stderr)
    answer, path, model, lat = _solve_logic_nl(ctx.client, prompt, category)
    if not (answer or "").strip():
        answer, lat = recover_empty_answer(ctx.client, prompt, category, model)
    answer = ensure_nonempty_answer(answer, category, prompt)
    ctx.metrics.log(task_id, category, path, model, lat)
    return {"task_id": task_id, "answer": answer}
