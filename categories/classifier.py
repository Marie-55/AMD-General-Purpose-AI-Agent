"""Regex-based task classifier: maps a prompt to one of the 8 competition
categories. Pure pattern matching, no model call -- classification has to
run before we know which local model to load, so it can't depend on one
being loaded yet.
"""
import re

from categories.task_categories import (
    FACTUAL_KNOWLEDGE,
    MATH_REASONING,
    SENTIMENT,
    SUMMARIZATION,
    NER,
    CODE_DEBUGGING,
    LOGIC_PUZZLE,
    CODE_GENERATION,
    ALL_CATEGORIES,
)

# Checked in priority order below -- more specific categories first, since
# e.g. a logic puzzle can easily mention people's names (looks NER-ish) or
# a code-debugging prompt can mention percentages (looks math-ish).
PATTERNS = {
    CODE_DEBUGGING: [
        r"```",
        r"\bbug\b",
        r"\bdebug\b",
        r"fix (the |this )?(function|code|bug)",
        r"find (the )?bug",
        r"corrected version",
        r"what('s| is) wrong with this",
        r"why does this (code|function) fail",
    ],
    CODE_GENERATION: [
        r"write a (python )?function",
        r"implement (a|the) (python )?function",
        r"write (a )?(python )?(script|program|class)",
        r"\bdef \w+\(",
        r"implement.*that (takes|returns|computes)",
    ],
    LOGIC_PUZZLE: [
        r"logic puzzle",
        r"each own[s]? a different",
        r"sit(s)? in a row",
        r"\bleft of\b",
        r"\bright of\b",
        r"unique solution",
        r"determine the (order|arrangement|position)",
        r"every clue",
        r"exactly one of",
        r"who owns (each|the)",
    ],
    NER: [
        r"named entit",
        r"extract.*entit",
        r"\bperson\b.*\borganization\b",
        r"\blocation\b.*\bdate\b",
        r"identify and label",
        r"extract.*as json",
        r"entity type",
    ],
    SUMMARIZATION: [
        r"summari[sz]e",
        r"\bsummary\b",
        r"\bcondense\b",
        r"in one sentence",
        r"in exactly \d+ words",
        r"no more than \d+ words",
        r"exactly \d+ (bullet|sentence)",
    ],
    SENTIMENT: [
        r"\bsentiment\b",
        r"positive.*negative",
        r"classify the sentiment",
        r"determine.*sentiment",
        r"label the sentiment",
    ],
    MATH_REASONING: [
        r"\bpercent(age)?\b",
        r"\d+\s*%",
        r"how much (change|remain)",
        r"average speed",
        r"\$\d",
        r"compounded",
        r"discount",
        r"investment",
        r"how many (remain|are left|units)",
    ],
    FACTUAL_KNOWLEDGE: [
        r"^what (is|are|causes|does)",
        r"^explain",
        r"^why (is|does|do|are)",
        r"^how (does|do|is|are)",
        r"explain (why|how)",
    ],
}

# Tie-break order when multiple categories score > 0 (most specific first).
PRIORITY = [
    CODE_DEBUGGING,
    CODE_GENERATION,
    LOGIC_PUZZLE,
    NER,
    SUMMARIZATION,
    SENTIMENT,
    MATH_REASONING,
    FACTUAL_KNOWLEDGE,
]


def classify(prompt: str) -> str:
    text = (prompt or "").lower()
    scores = {cat: 0 for cat in ALL_CATEGORIES}
    for cat, patterns in PATTERNS.items():
        for pat in patterns:
            if re.search(pat, text, re.IGNORECASE | re.MULTILINE):
                scores[cat] += 1

    best_cat, best_score = FACTUAL_KNOWLEDGE, 0
    for cat in PRIORITY:
        if scores[cat] > best_score:
            best_cat, best_score = cat, scores[cat]
    return best_cat
