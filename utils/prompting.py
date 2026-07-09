from __future__ import annotations

import re
from dataclasses import dataclass

from .parsing import normalize_text


WORD_RE = re.compile(r"\b[\w'-]+\b")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)

COMPRESSIBLE_FILLER = {
    "please",
    "kindly",
    "carefully",
    "simply",
    "just",
    "really",
    "very",
    "extremely",
    "clearly",
    "obviously",
    "basically",
    "actually",
    "in order to",
    "for your reference",
    "for context",
    "the following",
    "as follows",
}


@dataclass(frozen=True)
class PromptCompression:
    original_chars: int
    compressed_chars: int
    text: str

    @property
    def ratio(self) -> float:
        return self.compressed_chars / self.original_chars if self.original_chars else 1.0


def _strip_filler(text: str) -> str:
    result = text
    for phrase in sorted(COMPRESSIBLE_FILLER, key=len, reverse=True):
        result = re.sub(re.escape(phrase), "", result, flags=re.IGNORECASE)
    result = normalize_text(result)
    if not result:
        return text
    if len(result) > len(text) * 0.95:
        return text
    return result


def _score_sentence(sentence: str, task_hint: str | None = None) -> int:
    score = 0
    lowered = sentence.lower()
    words = WORD_RE.findall(sentence)
    if len(sentence) < 12:
        return score
    if "?" in sentence:
        score += 4
    if any(token in lowered for token in ("must", "exactly", "only", "return", "json", "code", "example")):
        score += 3
    if any(char.isdigit() for char in sentence):
        score += 2
    has_upper = any(word[:1].isupper() for word in words)
    has_lower = any(word[:1].islower() for word in words)
    if has_upper and has_lower:
        score += 1
    if task_hint:
        hint_tokens = {token for token in WORD_RE.findall(task_hint.lower()) if len(token) > 2}
        overlap = len({token.lower() for token in words} & hint_tokens)
        score += min(4, overlap)
    score += min(3, len(words) // 12)
    return score


def _budgeted_join(parts: list[str], budget_chars: int) -> str:
    collected: list[str] = []
    total = 0
    for part in parts:
        part = normalize_text(part)
        if not part:
            continue
        projected = total + len(part) + (1 if collected else 0)
        if projected > budget_chars:
            break
        collected.append(part)
        total = projected
    if collected:
        return " ".join(collected)
    return ""


def compress_prompt(
    text: str,
    *,
    task_hint: str | None = None,
    category: str | None = None,
    budget_chars: int = 900,
) -> PromptCompression:
    text = normalize_text(text)
    if task_hint is None and category is not None:
        task_hint = category
    if len(text) <= budget_chars:
        return PromptCompression(len(text), len(text), text)

    code_blocks = list(CODE_BLOCK_RE.finditer(text))
    if code_blocks:
        kept: list[str] = []
        for match in code_blocks[:2]:
            kept.append(match.group(0))
        remainder = CODE_BLOCK_RE.sub(" ", text)
        sentences = [item.strip() for item in SENTENCE_RE.split(remainder) if item.strip()]
        ranked = sorted(sentences, key=lambda sentence: _score_sentence(sentence, task_hint), reverse=True)
        kept.extend(ranked[: max(2, min(5, budget_chars // 180))])
        compressed = _budgeted_join(kept, budget_chars)
        if not compressed:
            compressed = text[:budget_chars].rstrip()
        return PromptCompression(len(text), len(compressed), compressed)

    paragraphs = [chunk.strip() for chunk in re.split(r"\n\s*\n", text) if chunk.strip()]
    if len(paragraphs) == 1:
        sentences = [item.strip() for item in SENTENCE_RE.split(paragraphs[0]) if item.strip()]
        if len(sentences) <= 2:
            compressed = _strip_filler(paragraphs[0])[:budget_chars].rstrip()
            return PromptCompression(len(text), len(compressed), compressed)
        ranked = sorted(sentences, key=lambda sentence: _score_sentence(sentence, task_hint), reverse=True)
        selected = ranked[: max(2, min(6, budget_chars // 140))]
        compressed = _budgeted_join(selected, budget_chars)
        if not compressed:
            compressed = _strip_filler(paragraphs[0])[:budget_chars].rstrip()
        return PromptCompression(len(text), len(compressed), compressed)

    scored_paragraphs = sorted(paragraphs, key=lambda paragraph: _score_sentence(paragraph, task_hint), reverse=True)
    selected: list[str] = []
    selected.extend(paragraphs[:1])
    if len(paragraphs) > 1:
        selected.append(paragraphs[-1])
    for paragraph in scored_paragraphs:
        if paragraph in selected:
            continue
        selected.append(paragraph)
        if len(_budgeted_join(selected, budget_chars)) >= budget_chars:
            break

    compressed = _budgeted_join(selected, budget_chars)
    if not compressed:
        compressed = _strip_filler(text)[:budget_chars].rstrip()
    if len(compressed) > budget_chars:
        compressed = compressed[:budget_chars].rstrip()
    return PromptCompression(len(text), len(compressed), compressed)


def compact_task_prompt(prompt: str, *, category: str, budget_chars: int = 900) -> str:
    hint = category.lower()
    compressed = compress_prompt(prompt, task_hint=hint, budget_chars=budget_chars)
    return compressed.text


def compact_user_prompt(instruction: str, prompt: str, *, category: str, budget_chars: int = 900) -> str:
    compressed_prompt = compact_task_prompt(prompt, category=category, budget_chars=budget_chars)
    return f"{instruction}\n\n{compressed_prompt}".strip()
