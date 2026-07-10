"""
Code-gen + local-execution path.

Two distinct execution strategies:

1. LOCAL MATH EXECUTION (zero Fireworks tokens)
    For math_reasoning: we first attempt to solve the problem by generating
    Python locally using the Qwen model (if loaded), OR by sending a single
    Fireworks call to get a short Python script, then running it in a sandboxed
    subprocess. The key difference from the old approach: we try a direct
    local Python eval pass first before touching Fireworks at all.

2. FIREWORKS CODE PATH (code_generation / code_debugging)
   For code tasks that need a code specialist: call Fireworks, extract code,
   verify via AST + exec, auto-fix if needed.

Logic puzzles are not handled here; they are routed through Fireworks NL.

Safety: every path has a fallback so no task ever returns empty.
"""
import os
import re
import subprocess
import sys
import tempfile
import json
from fractions import Fraction

# System prompt for math script generation — tight constraints so the
# model emits only what we need to execute, nothing extra.
CODE_GEN_SYSTEM = (
    "You are a Python code generator. Given a math problem, write a SELF-CONTAINED "
    "Python script using only the standard library (no external packages, no input()). "
    "Use the fractions.Fraction class for any division or fraction arithmetic to avoid "
    "floating-point errors. "
    "Assign the final answer to a variable named RESULT. "
    "If RESULT is a float or Fraction, convert to float and round to 2 decimal places: "
    "RESULT = round(float(RESULT), 2). "
    "If the result is a whole number after rounding (e.g. 600.0), print it as an integer: "
    "print(int(RESULT) if RESULT == int(RESULT) else RESULT). "
    "Return only one complete ```python code block and nothing else. "
    "No explanation, no markdown outside the block, and no extra text."
)


def build_codegen_messages(problem_text: str) -> list:
    return [
        {"role": "system", "content": CODE_GEN_SYSTEM},
        {"role": "user", "content": problem_text},
    ]


def extract_code(llm_output: str) -> str:
    """Extract the first ```python ... ``` block, or return the raw text."""
    m = re.search(r"```(?:python)?\s*(.*?)```", llm_output, flags=re.DOTALL)
    if m:
        return m.group(1).strip()

    text = llm_output.strip()
    if not text:
        return text

    if text.startswith("{"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return text

        for key in ("corrected_code", "code", "fixed_code"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        parts = parsed.get("corrected_parts")
        if isinstance(parts, list):
            for part in parts:
                if isinstance(part, dict):
                    for key in ("corrected_code", "code", "fix"):
                        value = part.get(key)
                        if isinstance(value, str) and value.strip():
                            return value.strip()

    return text


def _clean_numeric_output(raw: str) -> str:
    """
    Post-process a numeric result string to remove float noise.

    Examples:
      '599.9999999999999' → '600'
      '142.79999999999998' → '142.8'
      '20407.33'  → '20407.33'
      '16.0'      → '16'
    """
    raw = raw.strip()
    try:
        f = float(raw)
        rounded = round(f, 2)
        if rounded == int(rounded):
            return str(int(rounded))
        # Remove unnecessary trailing zeros after decimal
        return f"{rounded:.10g}"
    except ValueError:
        return raw


def run_code_safely(code: str, timeout_s: float = 8.0):
    """
    Execute *code* in an isolated subprocess with a hard wall-clock timeout.

    Returns ``(success: bool, output: str)``.  The output is cleaned of
    floating-point noise by ``_clean_numeric_output`` if it looks like a number.
    """
    path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            path = f.name
        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if proc.returncode != 0:
            return False, proc.stderr.strip()[-500:]
        out = proc.stdout.strip()
        if not out:
            return False, "empty output"
        # Clean numeric noise on the output.
        cleaned = _clean_numeric_output(out)
        return True, cleaned
    except subprocess.TimeoutExpired:
        return False, f"execution exceeded {timeout_s}s"
    except Exception as e:  # noqa: BLE001
        return False, str(e)
    finally:
        if path and os.path.exists(path):
            try:
                os.unlink(path)
            except OSError:
                pass
