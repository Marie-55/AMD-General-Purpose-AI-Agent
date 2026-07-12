"""Factual knowledge: local model first, escalate to Fireworks only if the
local answer is too short, vague, refused, or self-contradictory."""
import re
import sys

from categories.local_model import run_factual
from categories.normalizer import normalize_prompt, get_max_tokens
from categories.routing import route
from config import PER_REQUEST_TIMEOUT_S
from categories.handlers.base import (
    try_local_first,
    ensure_nonempty_answer,
    maybe_compress,
    recover_empty_answer,
    VAGUE_MARKERS,
)

_CONTRADICTION_RE = re.compile(
    # Patterns that signal self-contradiction in a factual answer.
    # Catches "does not guarantee X ... ensures X" and similar inversions.
    r"(does\s+not\s+(?:guarantee|ensure|provide|support|allow)\s+(\w+(?:\s+\w+){0,3}))"
    r".{0,120}"
    r"((?:ensures?|guarantees?|provides?|supports?|allows?)\s+\2)",
    re.IGNORECASE | re.DOTALL,
)


def _factual_answer_is_confident(answer: str) -> bool:
    """
    Return True only if the local model produced a substantive, internally
    consistent factual answer.

    Rejects:
    - empty / whitespace
    - fewer than 20 words (too terse)
    - known refusal / uncertainty phrases
    - self-contradictions: "does not guarantee X … ensures X"
    """
    if not (answer or "").strip():
        return False
    words = answer.split()
    if len(words) < 20:
        return False
    lowered = answer.lower()
    if any(m in lowered for m in VAGUE_MARKERS):
        return False
    # Catch internal contradictions (e.g. TCP t2 failure)
    if _CONTRADICTION_RE.search(answer):
        return False
    return True


def handle(task: dict, category: str, ctx) -> dict:
    task_id = task["task_id"]
    prompt = task["prompt"]

    local_answer = try_local_first(
        ctx, task_id, category, prompt,
        local_fn=run_factual,
        is_good_enough=_factual_answer_is_confident,
        path_name="local_factual",
    )
    if local_answer is not None:
        return {"task_id": task_id, "answer": local_answer}

    # Local answer is vague, too short, or refused → escalate to Fireworks
    print(f"[factual] {task_id}: local vague/empty, escalating to Fireworks", file=sys.stderr)
    fw_model = route(category)
    messages = normalize_prompt(prompt, category)
    messages = maybe_compress(messages, prompt)
    max_tok = get_max_tokens(category)
    fw_answer, fw_lat = ctx.client.call(fw_model, messages, max_tokens=max_tok,
                                         timeout=PER_REQUEST_TIMEOUT_S)
    fw_answer = (fw_answer or "").strip()
    if not fw_answer:
        fw_answer, fw_lat = recover_empty_answer(ctx.client, prompt, category, fw_model)
    fw_answer = ensure_nonempty_answer(fw_answer, category, prompt)
    ctx.metrics.log(task_id, category, "fireworks_factual", fw_model, fw_lat)
    return {"task_id": task_id, "answer": fw_answer}
