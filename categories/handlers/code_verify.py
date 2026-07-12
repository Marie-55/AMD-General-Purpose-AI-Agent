"""Code generation / debugging: Fireworks code specialist, AST/exec
verification, an auto-fix retry loop (see categories/code_verifier.py), and
a natural-language fallback if verification never succeeds."""
import ast
import json
import sys

from categories.code_exec import extract_code
from categories.code_verifier import verify_and_fix
from categories.normalizer import normalize_prompt, get_max_tokens
from categories.routing import route
from categories import prompts
from categories.task_categories import CODE_GENERATION, CODE_DEBUGGING
from config import PER_REQUEST_TIMEOUT_S
from categories.handlers.base import (
    ensure_nonempty_answer,
    extract_json_candidate,
    last_resort_answer,
    maybe_compress,
    nl_fallback,
    recover_empty_answer,
)


def _code_is_complete(code: str) -> bool:
    """Return True if the code contains at least one executable statement (not just docstring/comments)."""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Check if body has any non-docstring, non-pass statement
                for stmt in node.body:
                    if not isinstance(stmt, (ast.Expr, ast.Pass)):
                        return True
        return False
    except SyntaxError:
        return False


def _normalize_code_debug_answer(answer: str, prompt: str) -> str:
    """
    The model now returns clean markdown (## Bug Explanation / ## Corrected Code).
    Strip any leftover JSON wrapper if it accidentally appears, then return
    the answer as-is so the LLM judge sees human-readable markdown.
    """
    # If the model still returned a JSON blob despite the new instruction,
    # parse it and render it as markdown so we never return raw JSON.
    candidate = extract_json_candidate(answer, "{", "}")
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
            code_text = "\n\n".join(f"```python\n{s}\n```" for s in code_snippets) if code_snippets else ""
            return f"## Bug Explanation\n{issue_text}\n\n## Corrected Code\n{code_text}".strip()

    # Answer is already markdown (or plain text) — return directly.
    if not (answer or "").strip():
        return last_resort_answer(CODE_DEBUGGING, prompt)
    return answer.strip()


def _solve_via_code_verify(client, prompt: str, category: str, task_id: str = ""):
    """Generate code → verify (AST + exec) → auto-fix → NL fallback."""
    model = route(category)
    messages = normalize_prompt(prompt, category)
    messages = maybe_compress(messages, prompt)
    max_tok = get_max_tokens(category)
    llm_out, lat = client.call(model, messages, max_tokens=max_tok,
                               timeout=PER_REQUEST_TIMEOUT_S)
    if category == CODE_GENERATION:
        extracted = extract_code(llm_out or "")
        if not _code_is_complete(extracted):
            print(f"[code] {task_id}: incomplete code, retrying with higher budget", file=sys.stderr)
            # Force a retry with a stronger system prompt and double tokens
            retry_messages = [
                {"role": "system", "content": prompts.CODE_GENERATION_INCOMPLETE_RETRY_SYSTEM},
                {"role": "user", "content": prompt},
            ]
            llm_out, lat2 = client.call(model, retry_messages, max_tokens=8192, timeout=PER_REQUEST_TIMEOUT_S)
            lat.update(lat2)
            # update the extracted code for later verification

    # Detect reasoning-model truncation: code block started but got cut off
    if llm_out and "```" in llm_out:
        extracted = extract_code(llm_out)
        try:
            ast.parse(extracted)
            syntax_ok = True
        except SyntaxError:
            syntax_ok = False

        # Also catch the case where extract_code found no complete fence and
        # returned raw text that happens to parse (e.g. a comment + partial
        # docstring) but contains no actual function definition.
        has_def = "def " in extracted
        if not syntax_ok or not has_def:
            print(
                f"[code] {'syntax error' if not syntax_ok else 'no def found'} "
                f"— likely truncation, retrying with 2× token budget",
                file=sys.stderr,
            )
            expanded_tok = min(max_tok * 2, 6000)
            llm_out2, lat2 = client.call(model, messages, max_tokens=expanded_tok,
                                         timeout=PER_REQUEST_TIMEOUT_S)
            if llm_out2 and "```" in llm_out2:
                llm_out = llm_out2
                lat.update(lat2)

    if lat.get("error") or not llm_out:
        fb_answer, fb_lat = nl_fallback(client, prompt, category, model)
        return fb_answer, "code_verify_fallback", model, fb_lat

    # Guard against truncated output that has no code block at all.
    if "```" not in llm_out:
        print(f"[code] output has no code block, retrying with stricter prompt",
              file=sys.stderr)
        messages[0]["content"] = prompts.CODE_BLOCK_MISSING_RETRY_PREFIX + prompt[:200]
        llm_out, lat2 = client.call(model, messages, max_tokens=max_tok,
                                    timeout=PER_REQUEST_TIMEOUT_S)
        lat.update(lat2)
        if not llm_out or "```" not in llm_out:
            fb_answer, fb_lat = nl_fallback(client, prompt, category, model)
            return fb_answer, "code_verify_fallback", model, fb_lat

    final_answer, path, verify_lat = verify_and_fix(
        llm_output=llm_out, prompt=prompt, category=category,
        client=client, model=model, max_retries=2,
        timeout=PER_REQUEST_TIMEOUT_S,
    )
    lat.update(verify_lat)
    return final_answer or "", path, model, lat


def handle(task: dict, category: str, ctx) -> dict:
    task_id = task["task_id"]
    prompt = task["prompt"]

    answer, path, model, lat = _solve_via_code_verify(ctx.client, prompt, category, task_id)
    if category == CODE_DEBUGGING:
        answer = _normalize_code_debug_answer(answer, prompt)
    if not (answer or "").strip():
        answer, lat = recover_empty_answer(ctx.client, prompt, category, model)
    answer = ensure_nonempty_answer(answer, category, prompt)
    ctx.metrics.log(task_id, category, path, model, lat)
    return {"task_id": task_id, "answer": answer}
