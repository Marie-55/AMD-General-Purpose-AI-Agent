"""Sentiment: local model first, then Fireworks (Gemma tier, then a
minimax-forced strict retry if Gemma is empty or unlabeled)."""
import re
import sys

from categories.task_categories import SENTIMENT as CATEGORY
from categories.local_model import run_sentiment
from categories.normalizer import get_max_tokens
from categories.routing import route
from categories import prompts
from config import PER_REQUEST_TIMEOUT_S
from categories.handlers.base import try_local_first, ensure_nonempty_answer


def handle(task: dict, category: str, ctx) -> dict:
    task_id = task["task_id"]
    prompt = task["prompt"]

    local_answer = try_local_first(
        ctx, task_id, category, prompt,
        local_fn=run_sentiment,
        is_good_enough=lambda answer: bool(answer and answer.strip()),
        path_name="local_sentiment",
    )
    if local_answer is not None:
        return {"task_id": task_id, "answer": local_answer}

    # Local path failed → try Gemma (cheap_general tier-0), fall back to minimax
    print(f"[sentiment] {task_id}: local empty, escalating to Fireworks", file=sys.stderr)
    fw_model = route(category)  # resolves to gemma if up, else minimax
    sentiment_messages = [
        {"role": "system", "content": prompts.SENTIMENT_FIREWORKS_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    max_tok = get_max_tokens(category)
    fw_answer, fw_lat = ctx.client.call(
        fw_model, sentiment_messages, max_tokens=max_tok, timeout=PER_REQUEST_TIMEOUT_S
    )
    fw_answer = (fw_answer or "").strip()

    # If still empty or model returned garbage, force minimax with a strict prompt
    has_label = bool(re.match(r"(positive|negative|neutral)", fw_answer, re.IGNORECASE))
    if not fw_answer or not has_label:
        print(f"[sentiment] {task_id}: Gemma empty/bad, forcing minimax strict", file=sys.stderr)
        mm_model = route("reasoning_specialist")   # minimax always resolves
        strict_messages = [
            {"role": "system", "content": prompts.SENTIMENT_FIREWORKS_STRICT_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        fw_answer, fw_lat = ctx.client.call(
            mm_model, strict_messages, max_tokens=max_tok, timeout=PER_REQUEST_TIMEOUT_S
        )
        fw_answer = (fw_answer or "").strip()
        fw_model = mm_model

    fw_answer = ensure_nonempty_answer(fw_answer, category, prompt)
    ctx.metrics.log(task_id, category, "fireworks_sentiment", fw_model, fw_lat)
    return {"task_id": task_id, "answer": fw_answer}
