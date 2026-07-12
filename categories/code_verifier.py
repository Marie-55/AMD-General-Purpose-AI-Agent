"""
Code verification layer for code_generation and code_debugging tasks.

After the model returns a code-containing answer we attempt to verify it
before handing the result back to the caller:

  1. AST-parse the extracted code (catches syntax errors instantly, zero cost).
  2. For code_generation tasks: also *execute* the code in a sandboxed
     subprocess (reuses the harness already in code_exec).
  3. If verification fails and retries remain: ask the model to fix the broken
     code (one focused retry with a tight system prompt).
  4. If all retries are exhausted: fall back to a plain natural-language answer
     so the caller always gets *something* useful.

The tuple returned is ``(final_answer, path_taken, latency_metrics)`` where
``path_taken`` is one of:

  - ``"code_verified"``      — original output passed verification
  - ``"code_fixed_retry"``   — a retry produced a working fix
  - ``"code_fallback_nl"``   — all retries failed; natural-language fallback used
"""
import ast
import time

from categories.code_exec import extract_code, run_code_safely, run_code_generation_check
from config import CODE_FIX_MAX_TOKENS
from categories.prompts import (
    CODE_VERIFY_FALLBACK_SYSTEM as _FALLBACK_SYSTEM,
    CODE_VERIFY_DEFAULT_FALLBACK_SYSTEM as _DEFAULT_FALLBACK_SYSTEM,
    CODE_VERIFY_FIX_SYSTEM as _FIX_SYSTEM,
)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def ast_parse_ok(code: str) -> bool:
    """Return ``True`` if *code* can be parsed by the Python AST module."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


# ---------------------------------------------------------------------------
# Core verification logic
# ---------------------------------------------------------------------------

def verify_and_fix(
    llm_output: str,
    prompt: str,
    category: str,
    client,
    model: str,
    max_retries: int = 2,
    timeout: float = 25.0,
) -> tuple[str, str, dict]:
    """Verify *llm_output* and attempt fixes if verification fails.

    Parameters
    ----------
    llm_output:
        Raw text returned by the model (may contain a ```python block).
    prompt:
        The original user-facing task description (used in fix prompts).
    category:
        ``"code_generation"`` or ``"code_debugging"``.
    client:
        An object with a ``call(model, messages, max_tokens, timeout)`` method.
    model:
        Model identifier string forwarded to *client*.
    max_retries:
        How many fix attempts to make before giving up and returning a
        natural-language fallback answer.
    timeout:
        Wall-clock timeout in seconds for each model call during retries.

    Returns
    -------
    tuple[str, str, dict]
        ``(final_answer, path_taken, latency_metrics)``
    """
    latency: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Step (a): extract code from the raw LLM output.
    # ------------------------------------------------------------------
    code = extract_code(llm_output)

    # ------------------------------------------------------------------
    # Step (b): attempt AST parse.
    # ------------------------------------------------------------------
    syntax_ok = ast_parse_ok(code)

    # ------------------------------------------------------------------
    # Step (c): for code_generation, also run the code; for code_debugging
    #           a clean parse is sufficient verification.
    # ------------------------------------------------------------------
    if syntax_ok:
        if category == "code_generation":
            t0 = time.perf_counter()
            success, _run_output = run_code_generation_check(code)
            latency["exec_s"] = round(time.perf_counter() - t0, 3)
            if success:
                return llm_output, "code_verified", latency
            # Smoke test failed (or an actual call raised) — fall through
            # to the fix loop instead of trusting unexecuted code.
        else:
            # code_debugging: parse success is enough.
            return llm_output, "code_verified", latency

    # ------------------------------------------------------------------
    # Step (d): retry loop — ask the model to fix the broken code.
    # ------------------------------------------------------------------
    broken_code = code  # keep a reference to the code that failed
    for attempt in range(max_retries):
        fix_messages = [
            {"role": "system", "content": _FIX_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Original task: {prompt}\n\n"
                    f"Broken code:\n```python\n{broken_code}\n```\n\n"
                    "Fix it."
                ),
            },
        ]
        t0 = time.perf_counter()
        try:
            fixed_output, _fix_lat = client.call(
                model,
                fix_messages,
                max_tokens=CODE_FIX_MAX_TOKENS,
                timeout=timeout,
            )
        except Exception:  # noqa: BLE001
            latency[f"fix_attempt_{attempt}_s"] = round(time.perf_counter() - t0, 3)
            continue
        latency[f"fix_attempt_{attempt}_s"] = round(time.perf_counter() - t0, 3)

        if not fixed_output:
            continue

        fixed_code = extract_code(fixed_output)

        # Re-verify: AST parse is always required.
        if not ast_parse_ok(fixed_code):
            broken_code = fixed_code  # use latest broken code for next retry
            continue

        # For code_generation also verify by execution.
        if category == "code_generation":
            t1 = time.perf_counter()
            success, _out = run_code_generation_check(fixed_code)
            latency[f"fix_exec_{attempt}_s"] = round(time.perf_counter() - t1, 3)
            if not success:
                broken_code = fixed_code
                continue

        return fixed_output, "code_fixed_retry", latency

    # ------------------------------------------------------------------
    # Step (e): all retries exhausted — natural-language fallback.
    # ------------------------------------------------------------------
    fallback_system = _FALLBACK_SYSTEM.get(category, _DEFAULT_FALLBACK_SYSTEM)
    fallback_messages = [
        {"role": "system", "content": fallback_system},
        {"role": "user", "content": prompt},
    ]
    t0 = time.perf_counter()
    try:
        nl_answer, _nl_lat = client.call(
            model,
            fallback_messages,
            max_tokens=CODE_FIX_MAX_TOKENS,
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001
        nl_answer = f"[code_verifier] All verification attempts failed. Last error: {exc}"
    latency["fallback_s"] = round(time.perf_counter() - t0, 3)

    return nl_answer or "", "code_fallback_nl", latency
