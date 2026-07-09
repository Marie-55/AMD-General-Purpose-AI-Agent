from __future__ import annotations

from dataclasses import dataclass

from .models import ExecutionTier, TaskCategory


@dataclass(frozen=True)
class ExecutionPlan:
    tier: ExecutionTier
    reason: str
    prompt_chars: int
    sentence_count: int


REMOTE_ONLY_CATEGORIES = {
    TaskCategory.CODE.value,
    TaskCategory.DEBUGGING.value,
    TaskCategory.LOGIC.value,
}

LOCAL_RULES_CATEGORIES = {
    TaskCategory.SENTIMENT.value,
    TaskCategory.NER.value,
}


def _sentence_count(prompt: str) -> int:
    return max(1, sum(prompt.count(mark) for mark in ".!?"))


def _looks_like_code(prompt: str) -> bool:
    lowered = prompt.lower()
    return any(
        token in lowered
        for token in (
            "```",
            "def ",
            "class ",
            "import ",
            "return ",
            "function",
            "bug",
            "debug",
            "fix the code",
            "write code",
        )
    )


def choose_execution_plan(
    category: str,
    prompt: str,
    *,
    local_model_available: bool,
) -> ExecutionPlan:
    category = category.lower().strip()
    prompt = prompt.strip()
    chars = len(prompt)
    sentences = _sentence_count(prompt)
    code_like = _looks_like_code(prompt)

    if category in REMOTE_ONLY_CATEGORIES or code_like:
        return ExecutionPlan(
            tier=ExecutionTier.FIREWORKS,
            reason="code/logic tasks stay on Fireworks",
            prompt_chars=chars,
            sentence_count=sentences,
        )

    if category in LOCAL_RULES_CATEGORIES:
        return ExecutionPlan(
            tier=ExecutionTier.RULES,
            reason="structured extraction/classification is handled locally",
            prompt_chars=chars,
            sentence_count=sentences,
        )

    if category == TaskCategory.SUMMARY.value:
        if chars <= 1800 and sentences <= 8:
            return ExecutionPlan(
                tier=ExecutionTier.LOCAL_MODEL if local_model_available else ExecutionTier.RULES,
                reason="short summarization is cheap enough for local execution",
                prompt_chars=chars,
                sentence_count=sentences,
            )
        return ExecutionPlan(
            tier=ExecutionTier.FIREWORKS,
            reason="long summarization is routed to Fireworks to preserve local RAM",
            prompt_chars=chars,
            sentence_count=sentences,
        )

    if category == TaskCategory.FACTUAL.value:
        if chars <= 1200 and sentences <= 6:
            return ExecutionPlan(
                tier=ExecutionTier.LOCAL_MODEL if local_model_available else ExecutionTier.RULES,
                reason="short factual QA can be handled locally",
                prompt_chars=chars,
                sentence_count=sentences,
            )
        return ExecutionPlan(
            tier=ExecutionTier.FIREWORKS,
            reason="long factual QA is better handled by Fireworks",
            prompt_chars=chars,
            sentence_count=sentences,
        )

    if category == TaskCategory.MATH.value:
        if chars <= 500 and not code_like:
            return ExecutionPlan(
                tier=ExecutionTier.LOCAL_MODEL if local_model_available else ExecutionTier.RULES,
                reason="small math problems stay local",
                prompt_chars=chars,
                sentence_count=sentences,
            )
        return ExecutionPlan(
            tier=ExecutionTier.FIREWORKS,
            reason="larger math tasks go to Fireworks for better reasoning depth",
            prompt_chars=chars,
            sentence_count=sentences,
        )

    if category == TaskCategory.UNKNOWN.value:
        if chars <= 800 and not code_like:
            return ExecutionPlan(
                tier=ExecutionTier.LOCAL_MODEL if local_model_available else ExecutionTier.RULES,
                reason="short unknown prompts are cheap to try locally",
                prompt_chars=chars,
                sentence_count=sentences,
            )
        return ExecutionPlan(
            tier=ExecutionTier.FIREWORKS,
            reason="unclear or longer prompts are safer on Fireworks",
            prompt_chars=chars,
            sentence_count=sentences,
        )

    return ExecutionPlan(
        tier=ExecutionTier.FIREWORKS,
        reason="default to Fireworks for non-trivial reasoning",
        prompt_chars=chars,
        sentence_count=sentences,
    )

