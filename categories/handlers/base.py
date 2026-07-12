"""
Helpers shared by more than one category handler.

The biggest win here is try_local_first(): sentiment, summarization,
factual, and ner (see their respective handler modules) all used to
copy-paste the same "run local with a timeout, time it, check if the
answer is good enough, log it, return" block with small variations. That
duplication is exactly the kind of thing that made the factual_knowledge /
NER mix-up bug (main.py used to call the wrong local function for
factual_knowledge) hard to spot -- four near-identical blocks make a typo
in one of them blend in. Now there is exactly one implementation.
"""
import sys
import time

from categories.normalizer import normalize_prompt, get_max_tokens
from categories.prompt_compressor import compress_messages, estimate_tokens
from categories import prompts
from categories.task_categories import (
    SENTIMENT,
    NER,
    MATH_REASONING,
    CODE_DEBUGGING,
    CODE_GENERATION,
    LOGIC_PUZZLE,
)
from config import COMPRESS_THRESHOLD, PER_REQUEST_TIMEOUT_S

# Refusal / uncertainty phrases that disqualify a local-model answer from
# being accepted, shared by any category that gates a local answer on
# "does this actually look like a real answer" (currently factual and logic).
VAGUE_MARKERS = (
    "i don't know", "i do not know", "i'm not sure", "i am not sure",
    "i cannot", "i can't", "as an ai", "i don't have",
    "i am unable", "i'm unable", "no information",
    "i don't have enough", "i lack", "unclear",
)


def run_local_with_timeout(fn, *args):
    """
    Run fn(*args), returning '' on any failure (including a timeout).

    fn ultimately calls LocalModelSingleton.generate(), which bounds how
    long it waits for the model's lock (see generate()'s lock_timeout_s)
    and raises TimeoutError if it can't get a turn -- caught here like any
    other failure.

    There is deliberately no thread-pool-based timeout at this layer.
    Wrapping the call in `with ThreadPoolExecutor(...) as ex:` and catching
    `future.result(timeout=...)` does NOT free this thread after the
    timeout if the underlying call keeps running: exiting that `with` block
    still calls `shutdown(wait=True)`, which blocks until the background
    thread actually finishes, no matter how long that takes. That used to
    mean every "timed out" local call was silently discarded *after* paying
    its full real latency anyway -- the worst of both worlds. Bounding the
    wait inside generate() itself, at the one place that actually blocks
    (the lock acquisition), is what makes a timeout actually save time.
    """
    try:
        return fn(*args)
    except Exception:
        return ""


def try_local_first(
    ctx,
    task_id: str,
    category: str,
    prompt: str,
    local_fn,
    is_good_enough,
    path_name: str,
    model_name: str = "local_qwen2.5",
    postprocess=lambda answer, prompt: answer,
):
    """
    Attempt a local-model answer for *category*.

    Returns the (possibly post-processed) answer string and logs it via
    ctx.metrics if local_model is loaded, the model call itself succeeded,
    and is_good_enough(raw_answer) is True. Returns None so the caller can
    escalate to Fireworks otherwise.

    local_fn may return either a plain answer string, or a (answer, source)
    tuple where source is "llm" or "heuristic" (see local_model.run_sentiment
    / run_ner, the two functions with a built-in keyword-heuristic fallback
    that fires when the actual model call fails/times out). A
    heuristic-sourced answer is NEVER accepted here, regardless of
    is_good_enough -- it means the real model call never ran, and a
    regex/keyword guess is not an acceptable substitute for either a real
    local judgment or a Fireworks call, both of which are available and
    reliable. Accepting it here previously caused wrong NER labels and
    self-contradictory sentiment sentences to be reported as ordinary local
    successes. The heuristic remains available as the true last resort, in
    last_resort_answer() below, only if Fireworks escalation also fails.
    """
    local_model = ctx.local_model
    if local_model is None or not local_model.is_loaded():
        return None

    t0 = time.time()
    try:
        raw = run_local_with_timeout(local_fn, prompt, local_model)
    except Exception as exc:
        print(f"[{category}] {task_id} local failed: {exc}", file=sys.stderr)
        raw = ""
    elapsed_ms = round((time.time() - t0) * 1000, 1)

    if isinstance(raw, tuple):
        answer, source = raw
    else:
        answer, source = raw, "llm"

    if source == "heuristic":
        return None

    if not is_good_enough(answer):
        return None

    answer = postprocess(answer, prompt)
    lat = {
        "total_latency_ms": elapsed_ms,
        "ttft_ms": None, "prompt_tokens": None,
        "completion_tokens": None, "total_tokens": None, "error": None,
    }
    ctx.metrics.log(task_id, category, path_name, model_name, lat)
    return answer


def last_resort_answer(category: str, prompt: str) -> str:
    """Return a category-specific non-empty fallback if all other paths fail."""
    if category == SENTIMENT:
        # Never blindly return "neutral" — run the heuristic on the actual text
        # so we at least get the right polarity even when all model calls fail.
        try:
            from categories.local_model import _heuristic_sentiment_label, _heuristic_sentiment_sentence
            label = _heuristic_sentiment_label(prompt) or "neutral"
            return _heuristic_sentiment_sentence(label, prompt)
        except Exception:
            return "neutral"
    if category == NER:
        return "[]"
    if category == MATH_REASONING:
        return "0"
    if category == CODE_DEBUGGING:
        import json
        return json.dumps(
            {
                "issues": ["The code needs a correction, but I could not verify a safe fix."],
                "corrected_parts": [],
            },
            ensure_ascii=False,
        )
    if category == CODE_GENERATION:
        return "```python\npass\n```"
    if category == LOGIC_PUZZLE:
        return "I could not determine a unique solution."
    return "I could not determine a reliable answer."


def ensure_nonempty_answer(answer: str, category: str, prompt: str) -> str:
    return answer if (answer or "").strip() else last_resort_answer(category, prompt)


def extract_json_candidate(text: str, opening: str, closing: str) -> str:
    """Pull a JSON object/array substring out of *text*, preferring a fenced
    ```json block if present. Shared by the ner and code_verify handlers,
    both of which have to tolerate a model wrapping its JSON in prose or a
    markdown fence despite being told not to."""
    import re

    if not text:
        return ""

    fenced = re.search(r"```json\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
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


def maybe_compress(messages: list, prompt: str) -> list:
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


def recover_empty_answer(client, prompt: str, category: str, model: str):
    """Retry once with a stricter prompt when a task returns an empty answer."""
    messages = normalize_prompt(prompt, category)
    messages = maybe_compress(messages, prompt)

    messages[0]["content"] = messages[0]["content"] + prompts.RECOVER_EMPTY_ANSWER_EXTRA.get(
        category,
        prompts.RECOVER_EMPTY_ANSWER_DEFAULT_EXTRA,
    )

    max_tok = get_max_tokens(category)
    answer, lat = client.call(model, messages, max_tokens=max_tok, timeout=PER_REQUEST_TIMEOUT_S)
    return answer or "", lat


def nl_fallback(client, prompt: str, category: str, model: str):
    """Plain NL Fireworks call — last resort when all other paths fail."""
    messages = normalize_prompt(prompt, category)
    messages = maybe_compress(messages, prompt)
    max_tok = get_max_tokens(category)
    answer, lat = client.call(model, messages, max_tokens=max_tok, timeout=PER_REQUEST_TIMEOUT_S)
    return answer or "", lat
