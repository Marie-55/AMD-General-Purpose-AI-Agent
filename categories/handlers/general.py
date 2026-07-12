"""
Generic Fireworks path: normalize the prompt, call the model, retry once on
empty, guarantee a non-empty answer.

Every one of the 8 real categories has a dedicated handler (see
categories/handlers/__init__.py), so this only runs if classify() ever
returns something outside the known set -- a safety net against a crash,
not a path any current task should take.
"""
from categories.normalizer import normalize_prompt, get_max_tokens
from categories.routing import route
from config import PER_REQUEST_TIMEOUT_S
from categories.handlers.base import ensure_nonempty_answer, maybe_compress, recover_empty_answer


def handle(task: dict, category: str, ctx) -> dict:
    task_id = task["task_id"]
    prompt = task["prompt"]

    model = route(category)
    messages = normalize_prompt(prompt, category)
    messages = maybe_compress(messages, prompt)
    max_tok = get_max_tokens(category)
    answer, lat = ctx.client.call(model, messages, max_tokens=max_tok,
                                   timeout=PER_REQUEST_TIMEOUT_S)
    answer = answer or ""

    if not answer.strip():
        answer, lat = recover_empty_answer(ctx.client, prompt, category, model)
        answer = answer or ""

    if lat.get("error"):
        answer = answer or ""
    answer = ensure_nonempty_answer(answer, category, prompt)
    ctx.metrics.log(task_id, category, "direct_llm", model, lat)
    return {"task_id": task_id, "answer": answer}
