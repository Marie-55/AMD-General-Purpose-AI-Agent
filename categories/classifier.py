"""
Heuristic task classifier.

Deliberately NOT a model call: this is pure regex/keyword pattern matching that
runs in microseconds on CPU. It exists so we can route each incoming prompt to
the right category (and therefore the right model / execution path) WITHOUT
spending any Fireworks tokens and without invoking any model at all -- local
or hosted. This keeps classification cost at exactly zero, which matters
because only Fireworks-metered tokens count towards the token-efficiency score
and only Fireworks calls are "legal" scored inference.
"""
import re

# Patterns are checked in PRIORITY order (most specific / most distinctive first),
# so a prompt that matches several categories weakly still resolves sensibly
# (e.g. a summarization prompt that also contains numbers won't get misrouted to math).
CATEGORY_PATTERNS = {
    "code_debugging": [
        r"```",
        r"\bbug\b",
        r"\bhas a bug\b",
        r"corrected version",
        r"fix (the|this) (function|code|bug)",
    ],
    "code_generation": [
        r"write a (python )?function",
        r"implement (this|the following) in python",
        r"\bwrite code\b",
        r"\bdef \w+\(",
        r"implement this:",
    ],
    "logic_puzzle": [
        r"logic puzzle",
        r"constraint puzzle",
        r"every clue",
        r"unique solution",
        r"each own a different",
        r"sit in a row",
    ],
    "ner": [
        r"named entit",
        r"extract.*entit",
        r"label.*entit",
        r"person/org",
        r"person, organization, location",
    ],
    "summarization": [
        r"summari[sz]e",
        r"\bsummary\b",
        r"\bcondense\b",
        r"in one sentence",
        r"no more than \d+ words",
        r"2-sentence summary",
    ],
    "sentiment": [
        r"\bsentiment\b",
        r"positive/negative",
        r"classify the sentiment",
        r"label the sentiment",
    ],
    "math_reasoning": [
        r"how much change",
        r"average speed",
        r"solve step by step",
        r"work through the problem",
        r"population",
        r"\$\d",
        r"\bpercent(age)?\b",
    ],
    "factual_knowledge": [
        r"^what (is|causes|are)",
        r"^explain",
        r"who proposed",
        r"quick question",
        r"give a concise explanation",
    ],
}

# Tie-break order when multiple categories score equally (specific -> generic).
PRIORITY = [
    "code_debugging",
    "code_generation",
    "logic_puzzle",
    "ner",
    "summarization",
    "sentiment",
    "math_reasoning",
    "factual_knowledge",
]


def classify(prompt: str) -> str:
    """Return one of the 8 fixed category strings for a raw prompt."""
    text = prompt.lower()
    scores = {cat: 0 for cat in CATEGORY_PATTERNS}
    for cat, patterns in CATEGORY_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE):
                scores[cat] += 1

    best_cat, best_score = "factual_knowledge", 0
    for cat in PRIORITY:
        if scores[cat] > best_score:
            best_cat, best_score = cat, scores[cat]
    return best_cat