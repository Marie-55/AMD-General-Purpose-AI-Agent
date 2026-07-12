"""Math-reasoning solving with a self-consistency check.

Code that executes successfully can still encode the wrong arithmetic --
execution proves the code *ran*, not that its logic matches the word
problem. So the primary code-execution answer is cross-checked against an
independent natural-language re-derivation from the same model; on
agreement we trust the (execution-verified) code answer, on disagreement
we do one tie-breaking resample rather than silently trusting either side.
"""
import re

import config
from categories import prompts
from categories.code_exec import extract_code, run_code_safely


def _extract_number(text: str):
    if not text:
        return None
    m = re.search(r"answer:\s*\$?(-?[\d,]+\.?\d*)", text, re.IGNORECASE)
    candidate = m.group(1) if m else None
    if candidate is None:
        nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
        candidate = nums[-1] if nums else None
    if candidate is None:
        return None
    try:
        return float(candidate.replace(",", ""))
    except ValueError:
        return None


def _solve_via_code(prompt: str, runtime, temperature: float = 0.2):
    messages = [
        {"role": "system", "content": prompts.MATH_CODE_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    raw = runtime.generate(messages, max_tokens=config.MATH_CODE_MAX_TOKENS, temperature=temperature)
    code = extract_code(raw)
    if not code:
        return None, None

    ok, stdout, _stderr = run_code_safely(code, timeout=config.CODE_EXEC_TIMEOUT_S)
    if not ok or not stdout:
        return None, None

    value = _extract_number(stdout)
    if value is None:
        return None, None

    answer_text = f"{stdout}\n\nAnswer: {value:g}"
    return answer_text, value


def solve(prompt: str, runtime) -> str:
    code_answer, code_value = _solve_via_code(prompt, runtime)

    nl_messages = [
        {"role": "system", "content": prompts.MATH_NL_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    nl_answer = runtime.generate(nl_messages, max_tokens=config.MATH_NL_MAX_TOKENS, temperature=0.3)
    nl_value = _extract_number(nl_answer)

    if code_value is not None and nl_value is not None:
        if abs(code_value - nl_value) < 1e-6:
            return code_answer
        # Disagreement -- one tie-breaking resample of the code path.
        _, code_value_2 = _solve_via_code(prompt, runtime, temperature=0.6)
        if code_value_2 is not None and abs(code_value_2 - nl_value) < 1e-6:
            return nl_answer
        return code_answer  # default to the execution-verified answer

    if code_answer is not None:
        return code_answer
    if nl_answer and nl_answer.strip():
        return nl_answer
    return config.FALLBACK_ANSWER
