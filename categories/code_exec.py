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
import ast
import os
import re
import subprocess
import sys
import tempfile
import json
from fractions import Fraction

from categories.prompts import CODE_GEN_SYSTEM


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

    # No closing fence found. If there's an opening fence with nothing after
    # it to close it -- almost always a truncated generation (ran out of
    # max_tokens mid-block) -- strip the opening marker and return
    # everything after it, rather than falling through to the raw-text case
    # below. Returning the raw text here would keep the literal ``` marker
    # in the "code", which is guaranteed invalid Python: a real run hit
    # exactly this, executing a script whose first line was "```python" and
    # failing with SyntaxError before ever reaching the actual logic.
    m_open = re.search(r"```(?:python)?\s*", llm_output)
    if m_open:
        return llm_output[m_open.end():].strip()

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

# ---------------------------------------------------------------------------
# code_generation verification: actually CALL the function(s), don't just
# check for stdout. Most code_generation answers are pure function
# definitions with no top-level invocation, so "did the script print
# something" is always False for correct code — and never catches real bugs
# like using an unhashable dict as a set element, since a plain script run
# never actually calls the function.
# ---------------------------------------------------------------------------

_SMOKE_ARG_POOL = [
    "[{'a': 1}, {'b': 2}, {'a': 1}]",  # list of dicts — catches unhashable-in-set bugs
    "[3, 1, 2, 3, 1]",                 # list of ints — dup/sort/min/max bugs
    "['apple', 'banana', 'kiwi']",     # list of strings
    "'hello world'",                   # plain string
    "{'x': 1, 'y': 2}",                # dict
    "[1, 2, 3]",                       # secondary list arg (merges, etc.)
]


def _extract_function_signatures(code: str):
    """Best-effort list of (name, required_param_count, has_varargs) for every
    top-level function in *code*. Used only to build smoke-test calls."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    sigs = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            required = len(args.args) - len(args.defaults)
            sigs.append((node.name, max(required, 0), bool(args.vararg)))
    return sigs


def _build_smoke_test(code: str) -> str | None:
    """
    Return code + one try/except call per top-level function using generic
    dummy args, printing PASS:<name> or FAIL:<name>:<error>. Returns None if
    there's no top-level function to call (caller should fall back to a
    plain script run in that case).
    """
    sigs = _extract_function_signatures(code)
    if not sigs:
        return None

    calls = []
    for name, param_count, has_varargs in sigs:
        n = min(2, len(_SMOKE_ARG_POOL)) if has_varargs else min(param_count, len(_SMOKE_ARG_POOL))
        args = ", ".join(_SMOKE_ARG_POOL[:n]) if param_count > 0 or has_varargs else ""
        calls.append(
            f"try:\n"
            f"    {name}({args})\n"
            f"    print('PASS:{name}')\n"
            f"except Exception as e:\n"
            f"    print(f'FAIL:{name}:{{type(e).__name__}}:{{e}}')\n"
        )
    return code + "\n\n" + "\n".join(calls)


def run_code_generation_check(code: str, timeout_s: float = 8.0):
    """
    Verify a code_generation answer by calling every top-level function it
    defines with generic dummy inputs. Returns ``(success, detail)``.

    success is True only if the code parses, the smoke-test script runs
    without a crash, and no function raised (no "FAIL:" line in stdout).
    Falls back to the plain run_code_safely behaviour if there's no
    top-level function to call.
    """
    smoke_script = _build_smoke_test(code)
    if smoke_script is None:
        return run_code_safely(code, timeout_s=timeout_s)

    ok, output = run_code_safely(smoke_script, timeout_s=timeout_s)
    if not ok:
        return False, output
    if "FAIL:" in output:
        return False, output
    return True, output
