"""Extract a Python code block from LLM output and execute it safely in a
subprocess with a hard timeout.
"""
import re
import subprocess
import sys

_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    """Pull the first fenced code block out of *text*. Falls back to
    everything after an opening fence if the closing fence is missing
    (truncated generation), and to the raw text if there's no fence at all.
    """
    if not text:
        return ""
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    m_open = re.search(r"```(?:python)?\s*", text)
    if m_open:
        return text[m_open.end():].strip()
    return text.strip()


def run_code_safely(code: str, timeout: float = 8.0):
    """Execute *code* in a fresh subprocess. Returns (ok, stdout, stderr).

    A subprocess timeout is reliably killable, unlike trying to interrupt
    an in-process llama.cpp generation call -- that's why code execution
    (not model generation) is the one place a hard timeout actually stops
    the work rather than just giving up on waiting for it.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "", "execution timed out"
    except Exception as exc:
        return False, "", str(exc)
