from __future__ import annotations

from .models import RouteDecision, RuntimeConfig, TaskCategory


CATEGORY_ORDER = [
    TaskCategory.FACTUAL,
    TaskCategory.MATH,
    TaskCategory.SENTIMENT,
    TaskCategory.SUMMARY,
    TaskCategory.NER,
    TaskCategory.DEBUGGING,
    TaskCategory.LOGIC,
    TaskCategory.CODE,
]

CATEGORY_HINTS: dict[TaskCategory, tuple[str, ...]] = {
    TaskCategory.SUMMARY: ("summarise", "summarize", "summary", "condense", "one sentence", "shorten", "briefly"),
    TaskCategory.SENTIMENT: ("sentiment", "positive", "negative", "neutral", "opinion", "tone"),
    TaskCategory.NER: (
        "named entity",
        "entities",
        "extract all entities",
        "person",
        "organization",
        "organisation",
        "location",
        "date",
    ),
    TaskCategory.DEBUGGING: (
        "debug",
        "bug",
        "fix the code",
        "correct the code",
        "what is wrong",
        "traceback",
        "error in this code",
    ),
    TaskCategory.CODE: ("write a function", "implement", "code that", "generate code", "complete the function", "write code"),
    TaskCategory.LOGIC: ("constraint", "puzzle", "deductive", "must satisfy", "all conditions", "arrangement", "logic puzzle"),
    TaskCategory.MATH: ("calculate", "compute", "percentage", "how many", "word problem", "arithmetic", "projection", "multi-step"),
    TaskCategory.FACTUAL: ("what is", "what are", "define", "explain", "how does", "why does", "difference between"),
}

LOW_COST_ROLES = {"router", "factual", "sentiment", "summarization", "ner"}


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def classify_heuristically(prompt: str) -> RouteDecision:
    text = prompt.lower()
    scores: dict[TaskCategory, int] = {category: 0 for category in CATEGORY_ORDER}

    for category, hints in CATEGORY_HINTS.items():
        for hint in hints:
            if hint in text:
                scores[category] += 1 if len(hint) < 10 else 2

    if any(token in text for token in ("def ", "class ", "return ", "import ", "if __name__")):
        scores[TaskCategory.CODE] += 2
        scores[TaskCategory.DEBUGGING] += 1

    if any(token in text for token in ("prove", "must satisfy", "all conditions", "who is", "which order", "arrangement")):
        scores[TaskCategory.LOGIC] += 2

    if any(token in text for token in ("sum", "average", "percent", "percentage", "total", "profit", "loss", "ratio")):
        scores[TaskCategory.MATH] += 1

    best = max(scores, key=scores.get)
    best_score = scores[best]
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    second_score = ordered[1][1] if len(ordered) > 1 else 0

    if best_score == 0:
        return RouteDecision(TaskCategory.UNKNOWN.value, 0.0, "heuristic-none")

    confidence = min(0.95, 0.35 + 0.15 * best_score + 0.10 * max(0, best_score - second_score))
    source = "heuristic" if confidence >= 0.65 else "heuristic-ambiguous"
    return RouteDecision(best.value, confidence, source)


def rank_models(models: list[str], role: str) -> list[str]:
    role = role.lower()
    priorities = (
        [
            ("minimax", 0),
            ("kimi-k2p7-code", 1),
            ("kimi", 2),
            ("gemma-4-26b-a4b-it", 3),
            ("gemma-4-31b-it-nvfp4", 4),
            ("gemma-4-31b-it", 5),
        ]
        if role in LOW_COST_ROLES
        else [
            ("kimi-k2p7-code", 0),
            ("gemma-4-31b-it-nvfp4", 1),
            ("gemma-4-31b-it", 2),
            ("gemma-4-26b-a4b-it", 3),
            ("kimi", 4),
            ("minimax", 5),
        ]
    )

    def score(model: str) -> tuple[int, int, str]:
        lowered = model.lower()
        for needle, weight in priorities:
            if needle in lowered:
                return weight, len(lowered), model
        return 99, len(lowered), model

    return [model for _, _, model in sorted((score(model) for model in models), key=lambda item: (item[0], item[1], item[2]))]


def pick_model(config: RuntimeConfig, role: str) -> str:
    ranked = rank_models(config.allowed_models, role)
    if not ranked:
        raise RuntimeError("No allowed models are available.")
    return ranked[0]
