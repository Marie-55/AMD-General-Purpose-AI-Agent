"""NER: local model first (zero Fireworks tokens) if it returns a
non-empty, valid JSON array; otherwise escalate to Fireworks."""
import json
import sys

from categories.local_model import run_ner
from categories.normalizer import normalize_prompt, get_max_tokens
from categories.routing import route
from config import PER_REQUEST_TIMEOUT_S
from categories.handlers.base import (
    try_local_first,
    ensure_nonempty_answer,
    extract_json_candidate,
    maybe_compress,
    recover_empty_answer,
)


def _local_ner_is_valid(answer: str) -> bool:
    try:
        parsed = json.loads(answer or "[]")
    except (json.JSONDecodeError, ValueError):
        return False
    return isinstance(parsed, list) and len(parsed) > 0


def _normalize_ner_answer(answer: str, prompt: str) -> str:
    candidate = extract_json_candidate(answer, "[", "]")
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
                # Only require non-empty text and a non-empty type — no whitelist.
                if not text_value or not type_value:
                    continue
                key = (text_value, type_value)
                if key in seen:
                    continue
                seen.add(key)
                normalized.append({"text": text_value, "type": type_value})
            return json.dumps(normalized, ensure_ascii=False)
    return "[]"


def handle(task: dict, category: str, ctx) -> dict:
    task_id = task["task_id"]
    prompt = task["prompt"]

    local_ner = try_local_first(
        ctx, task_id, category, prompt,
        local_fn=run_ner,
        is_good_enough=_local_ner_is_valid,
        path_name="local_ner",
    )
    if local_ner is not None:
        return {"task_id": task_id, "answer": local_ner}

    print(f"[ner] {task_id}: local empty/invalid, escalating to Fireworks", file=sys.stderr)
    fw_model = route(category)
    fw_messages = normalize_prompt(prompt, category)
    fw_messages = maybe_compress(fw_messages, prompt)
    max_tok = get_max_tokens(category)
    fw_answer, fw_lat = ctx.client.call(fw_model, fw_messages, max_tokens=max_tok,
                                         timeout=PER_REQUEST_TIMEOUT_S)
    fw_answer = _normalize_ner_answer(fw_answer or "", prompt)
    if not fw_answer or fw_answer == "[]":
        fw_answer, fw_lat = recover_empty_answer(ctx.client, prompt, category, fw_model)
        fw_answer = _normalize_ner_answer(fw_answer or "", prompt)
    fw_answer = ensure_nonempty_answer(fw_answer, category, prompt)
    ctx.metrics.log(task_id, category, "fireworks_ner", fw_model, fw_lat)
    return {"task_id": task_id, "answer": fw_answer}
