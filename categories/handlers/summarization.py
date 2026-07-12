"""Summarization: local model first, escalating to Fireworks only if the
local model produced nothing usable at all -- a short answer is not by
itself a reason to distrust it: many summarization prompts explicitly ask
for a short form ("in one sentence", "in exactly 5 words"), so a correctly
terse local answer must not be punished for being terse. run_summarization()
already rejects incomplete/truncated output internally (returns "" when the
result doesn't look like a properly finished sentence), so by the time this
gate runs, what's left to check is just "did we get anything at all."""
import sys

from categories.local_model import run_summarization
from categories.normalizer import get_max_tokens, enforce_summarization_constraint
from categories.routing import route
from categories import prompts
from config import PER_REQUEST_TIMEOUT_S
from categories.handlers.base import try_local_first, ensure_nonempty_answer


def handle(task: dict, category: str, ctx) -> dict:
    task_id = task["task_id"]
    prompt = task["prompt"]

    local_answer = try_local_first(
        ctx, task_id, category, prompt,
        local_fn=run_summarization,
        is_good_enough=lambda answer: bool(answer and answer.strip()),
        postprocess=enforce_summarization_constraint,
        path_name="local_summarization",
    )
    if local_answer is not None:
        return {"task_id": task_id, "answer": local_answer}

    # Fallback to Fireworks with higher max_tokens and stricter prompt
    print(f"[summarization] {task_id}: local empty, escalating to Fireworks", file=sys.stderr)
    fw_model = route(category)   # minimax
    max_tok = get_max_tokens(category)   # now 800

    strict_messages = [
        {"role": "system", "content": prompts.SUMMARIZATION_FIREWORKS_STRICT_SYSTEM},
        {
            "role": "user",
            "content": prompts.SUMMARIZATION_FIREWORKS_USER_TEMPLATE.format(prompt=prompt),
        },
    ]

    answer, lat = ctx.client.call(
        fw_model, strict_messages,
        max_tokens=max_tok,
        timeout=PER_REQUEST_TIMEOUT_S,
    )
    answer = (answer or "").strip()

    # If still empty, one retry with even higher budget and no prefix
    if not answer or lat.get("error"):
        print(f"[summarization] {task_id}: minimax empty, retrying with higher budget", file=sys.stderr)
        retry_messages = [
            {"role": "system", "content": prompts.SUMMARIZATION_FIREWORKS_RETRY_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        answer, lat = ctx.client.call(
            fw_model, retry_messages,
            max_tokens=2200,   # extra headroom above the primary attempt's budget
            timeout=PER_REQUEST_TIMEOUT_S,
        )
        answer = (answer or "").strip()

    answer = enforce_summarization_constraint(answer, prompt)
    answer = ensure_nonempty_answer(answer, category, prompt)
    ctx.metrics.log(task_id, category, "summarization_fw_fallback", fw_model, lat)
    return {"task_id": task_id, "answer": answer}
