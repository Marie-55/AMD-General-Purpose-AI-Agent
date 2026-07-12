"""
Math reasoning: local Python code generation + execution first (zero
Fireworks tokens), Fireworks code generation as fallback, plain NL as the
last resort. Explanations (when the prompt asks for steps) are appended
around the same numeric answer regardless of which path produced it.
"""
import re
import sys

from categories.code_exec import build_codegen_messages, extract_code, run_code_safely
from categories.normalizer import normalize_prompt, get_max_tokens
from categories.routing import route
from categories import prompts
from config import PER_REQUEST_TIMEOUT_S, CODE_EXEC_TIMEOUT_S, MATH_CODEGEN_MAX_TOKENS
from categories.handlers.base import (
    ensure_nonempty_answer,
    maybe_compress,
    recover_empty_answer,
)

# ---------------------------------------------------------------------------
# Detect if the prompt asks for reasoning/steps
# ---------------------------------------------------------------------------
_REASONING_KEYWORDS = re.compile(
    r"\b(show|explain|step(?:s)?|reasoning|walk(?:[-\s]?through)?|how|why|work(?:[-\s]?out)?)"
    r"|step[-\s]by[-\s]step",
    re.IGNORECASE,
)


def _prompt_wants_steps(prompt: str) -> bool:
    """Return True if the prompt explicitly asks for steps or explanation."""
    return bool(_REASONING_KEYWORDS.search(prompt))


def _split_work_and_final(output: str) -> tuple[str, str]:
    """Split captured stdout into (intermediate work lines, final value).

    run_code_safely leaves multi-line stdout untouched (its numeric cleanup
    only fires when the whole output parses as one float), so when the math
    codegen prompt's requested intermediate print() calls actually ran,
    `output` here is exactly what the script printed: zero or more work
    lines followed by the final RESULT on the last line.
    """
    lines = [l.strip() for l in output.splitlines() if l.strip()]
    if not lines:
        return "", output
    return "; ".join(lines[:-1]), lines[-1]


def _format_shown_work(output: str) -> str:
    """
    Build a compact 'shown work' answer from captured stdout.

    judge_guide.md's math rubric expects "minor arithmetic shown or
    implied" even for prompts that don't explicitly ask to show steps --
    the codegen prompts now print each intermediate value for exactly this
    reason, so this is free (no extra model call): it just formats stdout
    that was already captured.
    """
    work, final_value = _split_work_and_final(output)
    if work:
        return f"{work}. Answer: {final_value}"
    return f"Answer: {final_value}"


def _solve_math(client, prompt: str, category: str, local_model=None):
    model_used = "local_qwen2.5-3b"  # will be overwritten if fallback used

    # ------------------------------------------------------------------
    # 1. LOCAL CODE GENERATION + EXECUTION
    # ------------------------------------------------------------------
    local_code = ""
    if local_model and local_model.is_loaded():
        try:
            from categories.local_model import run_math_code
            local_code = run_math_code(prompt, local_model)
        except Exception:
            pass
    print(f"[math] local_code length: {len(local_code)}")
    code = extract_code(local_code) if local_code else ""
    print(f"[math] extracted code length: {len(code)}")
    # Safety net: if local model forgot the print() call, inject one on the last assignment.
    # run_code_safely treats empty stdout as failure, so this is the #1 reason local math fails.
    if code and "print(" not in code:
        lines = [l for l in code.splitlines() if l.strip() and not l.strip().startswith("#")]
        last_var = None
        for line in reversed(lines):
            m = re.match(r'^([A-Za-z_]\w*)\s*=', line.strip())
            if m:
                last_var = m.group(1)
                break
        if last_var:
            code = code.rstrip() + f"\nprint({last_var})"
            print(f"[math] injected print({last_var}) — model forgot to print", file=sys.stderr)
        else:
            # No assignment found; wrap the whole block in an exec+capture pattern
            code = code.rstrip() + "\n# no printable variable found"
    ok, output = (False, "")
    if code:
        ok, output = run_code_safely(code, timeout_s=CODE_EXEC_TIMEOUT_S)

    # If local exec succeeded, try local explanation if requested
    if ok and output:
        if _prompt_wants_steps(prompt):
            explanation = ""
            if local_model and local_model.is_loaded():
                try:
                    from categories.local_model import run_math_explanation
                    explanation = run_math_explanation(prompt, local_model)
                except Exception:
                    pass
            _, final_value = _split_work_and_final(output)
            if explanation and len(explanation.split()) >= 20:
                # Ensure the explanation ends with "Answer:" and numeric value
                # but we already have the numeric output from execution,
                # so we can just append it if missing.
                if "answer:" not in explanation.lower():
                    explanation = f"{explanation}\n\nAnswer: {final_value}"
                return explanation, "math_local_with_explanation", model_used, {}
            else:
                # Local explanation failed or too short – fall back to Fireworks explanation
                exp_messages = normalize_prompt(prompt, category)
                exp_messages = maybe_compress(exp_messages, prompt)
                fw_model = route(category)
                exp_out, _ = client.call(fw_model, exp_messages,
                                         max_tokens=get_max_tokens(category),
                                         timeout=PER_REQUEST_TIMEOUT_S)
                if exp_out and exp_out.strip():
                    exp_clean = re.sub(r"\n*Answer:.*\Z", "", exp_out.strip(),
                                       flags=re.IGNORECASE | re.DOTALL).strip()
                    combined = f"{exp_clean}\n\nAnswer: {final_value}"
                    return combined, "math_local_exec_with_fw_explanation", fw_model, {}
                # If even FW explanation fails, fall back to the printed work trail
                return _format_shown_work(output), "math_local_exec_no_explanation", model_used, {}
        return _format_shown_work(output), "math_local_exec", model_used, {}

    # ------------------------------------------------------------------
    # 2. LOCAL CODE FAILED – FALLBACK TO FIREWORKS CODE GENERATION
    # ------------------------------------------------------------------
    fw_model = route(category)
    messages = build_codegen_messages(prompt)
    llm_out, lat = client.call(fw_model, messages, max_tokens=MATH_CODEGEN_MAX_TOKENS,
                               timeout=PER_REQUEST_TIMEOUT_S)
    code = extract_code(llm_out) if (llm_out and not lat.get("error")) else ""
    print(f"[math] fw_code length: {len(llm_out or '')} (error={lat.get('error')})", file=sys.stderr)
    print(f"[math] fw_extracted code length: {len(code)}", file=sys.stderr)
    ok, output = (False, "")
    if code:
        ok, output = run_code_safely(code, timeout_s=CODE_EXEC_TIMEOUT_S)
        if not ok:
            print(f"[math] fw_code execution failed: {output[:200]!r}", file=sys.stderr)

    # Retry with fix prompt if first attempt failed
    if not (ok and output) and code:
        fix_messages = [
            {"role": "system", "content": prompts.CODE_GEN_SYSTEM},
            {"role": "user", "content": (
                f"{prompt}\n\nYour previous script failed with this error:\n{output}\n\n"
                "Fix the script and return only the corrected ```python block."
            )},
        ]
        fix_out, fix_lat = client.call(fw_model, fix_messages, max_tokens=MATH_CODEGEN_MAX_TOKENS,
                                        timeout=PER_REQUEST_TIMEOUT_S)
        if fix_out and not fix_lat.get("error"):
            fix_code = extract_code(fix_out)
            if fix_code:
                ok, output = run_code_safely(fix_code, timeout_s=CODE_EXEC_TIMEOUT_S)
        lat.update({f"fix_{k}": v for k, v in fix_lat.items() if k not in lat})
        if not ok:
            print(f"[math] fw_code fix-retry also failed: {output[:200]!r}", file=sys.stderr)

    if ok and output:
        # If we have a successful numeric answer, handle explanation (FW)
        if _prompt_wants_steps(prompt):
            exp_messages = normalize_prompt(prompt, category)
            exp_messages = maybe_compress(exp_messages, prompt)
            exp_out, _ = client.call(fw_model, exp_messages,
                                     max_tokens=get_max_tokens(category),
                                     timeout=PER_REQUEST_TIMEOUT_S)
            if exp_out and exp_out.strip():
                exp_clean = re.sub(r"\n*Answer:.*\Z", "", exp_out.strip(),
                                   flags=re.IGNORECASE | re.DOTALL).strip()
                _, final_value = _split_work_and_final(output)
                combined = f"{exp_clean}\n\nAnswer: {final_value}"
                return combined, "math_fw_code_with_explanation", fw_model, lat
        return _format_shown_work(output), "math_fw_code", fw_model, lat

    # Both code attempts failed – final NL fallback. This is the one path
    # with no code-execution result to verify against, so it uses a
    # dedicated prompt that explicitly asks for a self-check pass instead of
    # the generic math instruction (see prompts.py::MATH_NL_FALLBACK_VERIFIED_SYSTEM).
    print("[math] both code attempts failed, falling back to verified NL reasoning", file=sys.stderr)
    if _prompt_wants_steps(prompt):
        verified_messages = [
            {"role": "system", "content": prompts.MATH_NL_FALLBACK_VERIFIED_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        fb_answer, fb_lat = client.call(fw_model, verified_messages,
                                        max_tokens=get_max_tokens(category),
                                        timeout=PER_REQUEST_TIMEOUT_S)
    else:
        terse_messages = [
            {"role": "system", "content": prompts.MATH_TERSE_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        fb_answer, fb_lat = client.call(fw_model, terse_messages, max_tokens=250,
                                        timeout=PER_REQUEST_TIMEOUT_S)
        fb_answer = fb_answer or ""
    return fb_answer, "math_nl_fallback", fw_model, fb_lat


def handle(task: dict, category: str, ctx) -> dict:
    task_id = task["task_id"]
    prompt = task["prompt"]

    answer, path, model, lat = _solve_math(ctx.client, prompt, category, ctx.local_model)
    if not (answer or "").strip():
        answer, lat = recover_empty_answer(ctx.client, prompt, category, model)
    answer = ensure_nonempty_answer(answer, category, prompt)
    ctx.metrics.log(task_id, category, path, model, lat)
    return {"task_id": task_id, "answer": answer}
