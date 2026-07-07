from __future__ import annotations

import json
import re
from typing import Any


JSON_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
CODE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def strip_code_fences(text: str) -> str:
    match = CODE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def compact_text(text: str) -> str:
    return normalize_text(text).strip('"').strip("'").strip()


def extract_json_candidate(text: str) -> str | None:
    match = JSON_RE.search(text)
    if match:
        return match.group(1).strip()

    first_obj = text.find("{")
    last_obj = text.rfind("}")
    if first_obj != -1 and last_obj != -1 and last_obj > first_obj:
        return text[first_obj : last_obj + 1].strip()

    first_arr = text.find("[")
    last_arr = text.rfind("]")
    if first_arr != -1 and last_arr != -1 and last_arr > first_arr:
        return text[first_arr : last_arr + 1].strip()

    return None


def parse_json_loose(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        candidate = extract_json_candidate(text)
        if candidate is None:
            raise
        return json.loads(candidate)


def summarization_budget(prompt: str) -> int:
    lower = prompt.lower()
    if "one sentence" in lower or "one-sentence" in lower:
        return 80
    word_match = re.search(r"(\d+)\s*words?", lower)
    if word_match:
        words = int(word_match.group(1))
        return max(64, min(256, words * 3))
    char_match = re.search(r"(\d+)\s*characters?", lower)
    if char_match:
        chars = int(char_match.group(1))
        return max(64, min(256, chars // 2))
    return 128


def sanitize_final_answer(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    if stripped.startswith("```"):
        stripped = strip_code_fences(stripped)
    lines = [line.rstrip() for line in stripped.splitlines() if line.strip()]
    if len(lines) > 1 and not stripped.lstrip().startswith("{") and not stripped.lstrip().startswith("["):
        return lines[-1].strip()
    return stripped.strip()


def collapse_json_answer(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))
