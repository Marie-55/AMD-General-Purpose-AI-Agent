"""
Code-gen + local-execution path.

For math_reasoning and logic_puzzle tasks, instead of asking the model to
reason the whole answer out in natural language (expensive completion
tokens, and arithmetic/constraint mistakes are common), we ask it to write a
SHORT Python script that computes the answer, then we execute that script
locally (CPU -- these are lightweight symbolic/arithmetic workloads, no GPU
needed) in an isolated subprocess with a hard timeout.

This buys us two things:
  1. Higher accuracy: arithmetic and constraint search are solved exactly by
     real code instead of approximated by token generation.
  2. Fewer tokens: the model only has to emit a short script, not a long
     chain-of-thought explanation -- and we never pay for "thinking" tokens
     that the local interpreter can do for free.

Safety net: if code generation, extraction, or execution fails or times out,
we fall back to a direct natural-language answer from the same model
(see orchestrator._direct_fallback), so a single bad generation never
produces a missing/malformed answer.
"""
import os
import re
import subprocess
import sys
import tempfile

CODE_GEN_SYSTEM = (
    "You are a Python code generator. Given a problem, write a SELF-CONTAINED Python "
    "script (standard library only, no input(), no external packages) that computes the "
    "answer and assigns the final result to a variable named RESULT (a string or number), "
    "then ends with exactly one line: print(RESULT). Do not print anything else. "
    "Return ONLY a single ```python code block -- no explanation before or after."
)


def build_codegen_messages(problem_text: str) -> list:
    return [
        {"role": "system", "content": CODE_GEN_SYSTEM},
        {"role": "user", "content": problem_text},
    ]


def extract_code(llm_output: str) -> str:
    m = re.search(r"```(?:python)?\s*(.*?)```", llm_output, flags=re.DOTALL)
    return m.group(1).strip() if m else llm_output.strip()


def run_code_safely(code: str, timeout_s: float = 8.0):
    """
    Executes generated code in an isolated subprocess (separate process from the
    orchestrator/harness) with a hard wall-clock timeout well under the per-task
    budget. Returns (success: bool, output_or_error: str).
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
        return (True, out) if out else (False, "empty output")
    except subprocess.TimeoutExpired:
        return False, f"execution exceeded {timeout_s}s"
    except Exception as e:  # noqa: BLE001 - want to catch anything and fall back gracefully
        return False, str(e)
    finally:
        if path and os.path.exists(path):
            try:
                os.unlink(path)
            except OSError:
                pass