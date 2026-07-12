"""
Heuristic task classifier with optional local-model second-pass.

Two-pass strategy
-----------------
1. ``classify_regex`` scores every pattern and returns ``(category, score)``.
2. ``classify`` decides how much to trust that score:
   - score >= 2  → high confidence, return immediately.
   - score == 1  → weak signal; local model confirms or overrides.
   - score == 0  → no signal; local model decides; fallback = factual_knowledge.

NER guard
---------
Qwen2.5 tends to classify any prompt that mentions people's names as ``ner``
even when the prompt is actually a logic puzzle or factual question.  After
classification we apply ``_ner_is_plausible()`` which checks whether the
prompt contains explicit NER task framing (extract, identify, label, PERSON,
ORG …).  If that check fails, the ``ner`` label is rejected and the next-best
category (or factual_knowledge) is used instead.
"""
import re
from typing import Optional, Tuple

from categories.task_categories import (
    CODE_DEBUGGING,
    CODE_GENERATION,
    LOGIC_PUZZLE,
    NER,
    SUMMARIZATION,
    SENTIMENT,
    MATH_REASONING,
    FACTUAL_KNOWLEDGE,
)

# ---------------------------------------------------------------------------
# Patterns — checked in PRIORITY order (most specific first)
# ---------------------------------------------------------------------------

CATEGORY_PATTERNS = {
    CODE_DEBUGGING: [
        r"```",
        r"\bbug\b",
        r"\bhas a bug\b",
        r"corrected version",
        r"fix (the|this) (function|code|bug)",
        r"\bdebug\b",
        r"find (the |and fix )?the bug",
        r"identify.*bug",
    ],
    CODE_GENERATION: [
        r"write a (python )?function",
        r"implement (this|the following) in python",
        r"\bwrite code\b",
        r"\bdef \w+\(",
        r"implement this:",
        r"implement a (python )?function",
        r"write a (python )?(script|program|class)",
        r"implement.*function that",
    ],
    LOGIC_PUZZLE: [
        r"logic puzzle",
        r"constraint puzzle",
        r"every clue",
        r"unique solution",
        r"each own a different",
        r"sit in a row",
        # Spatial / ordering clues
        r"\bleft of\b",
        r"\bright of\b",
        r"\bimmediately (left|right)\b",
        r"\bimmediately (above|below)\b",
        r"\bfinishes? before\b",
        r"\bfinishing order\b",
        r"\barrangement\b",
        r"\boffices?\s+\d",
        r"\bseats?\s+\d",
        r"\bstacked\b",
        r"\babove the\b",
        r"\bbelow the\b",
        r"\borders?\b.*\b(first|second|third|fourth)\b",
        r"\bdetermine (the |each )?(order|arrangement|seat|position|office)",
        r"\bwho owns (each|the)\b",
        r"\beach own\b",
        r"\bexactly one of\b",
        # Constraint deduction keywords
        r"\bnot in (seat|office|position)\b",
        r"\bnot (last|first)\b",
        r"\bnone of them\b",
        r"every label is wrong",
        r"all labels? (are |is )wrong",
        r"inspect (one|a) (fruit|item|object)",
        r"which (box|door|container).*(choose|pick|open|inspect)",
        r"without looking",
    ],
    NER: [
        r"named entit",
        r"extract.*entit",
        r"label.*entit",
        r"person/org",
        r"person,\s*organization,\s*location",
        r"identify.*\b(PERSON|ORG|LOCATION|DATE)\b",
        r"extract.*\b(PERSON|ORG|LOCATION|DATE)\b",
        r"\bPERSON\b.*\bORG\b",
        r"\bORG\b.*\bLOCATION\b",
        r"return.*as json.*entit",
        r"extract every\b",
        r"label every\b.*entit",
        r"grouped by type",
        r"identify and label every",
    ],
    SUMMARIZATION: [
        r"summari[sz]e",
        r"\bsummary\b",
        r"\bcondense\b",
        r"in one sentence",
        r"no more than \d+ words",
        r"2-sentence summary",
        r"single.sentence summary",
        r"in exactly \d+ words",
        r"exactly \d+ (bullet|sentence)",
        r"as (exactly )?\d+ bullet",
        r"as exactly \d+",
    ],
    SENTIMENT: [
        r"\bsentiment\b",
        r"positive/negative",
        r"classify the sentiment",
        r"label the sentiment",
        r"determine the (overall )?sentiment",
        r"determine.*sentiment",
        r"classify.*sentiment",
        r"\bpositive\b.*\bnegative\b",
    ],
    MATH_REASONING: [
        r"how much (change|is|are|remain)",
        r"average speed",
        r"solve step by step",
        r"work through the problem",
        r"\bpopulation\b",
        r"\$\d",
        r"\bpercent(age)?\b",
        r"what is (the )?(final price|total|sum|difference|average|capacity|worth)",
        r"how many remain",
        r"compounded (annually|monthly)",
        r"\b\d+\s*%\b",
        r"(full|fills?)\b.*(reservoir|tank|container)",
        r"(grows?|increases?)\s+by\s+\d+\s*%",
        r"originally costs?",
        r"discount",
        r"investment",
        r"annually",
    ],
    FACTUAL_KNOWLEDGE: [
        r"^what (is|causes|are|does)",
        r"^explain",
        r"who proposed",
        r"quick question",
        r"give a concise explanation",
        r"^why (is|does|do|are|can)",
        r"^how (does|do|is|are)",
        r"explain why",
        r"explain how",
    ],
}

# Tie-break order (specific → generic).
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

# ---------------------------------------------------------------------------
# NER plausibility guard
# ---------------------------------------------------------------------------
# Keywords that must appear in a prompt for it to be a genuine NER task.
# Without at least one of these, "ner" is almost certainly a false positive
# caused by the presence of proper nouns / people names.
_NER_REQUIRED_RE = re.compile(
    r"named entit|extract.*entit|label.*entit|identify.*entit|"
    r"person/org|PERSON|ORG|LOCATION|DATE|"
    r"extract every|label every|grouped by type|"
    r"identify and label|return.*as json|"
    r"extract.*as json|entity type",
    re.IGNORECASE,
)


def _ner_is_plausible(prompt: str) -> bool:
    """Return True only if the prompt explicitly asks for NER extraction."""
    return bool(_NER_REQUIRED_RE.search(prompt))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_categories(prompt: str) -> dict:
    """Score every category against *prompt*. Shared by classify_regex and classify()."""
    text = prompt.lower()
    scores = {cat: 0 for cat in CATEGORY_PATTERNS}
    for cat, patterns in CATEGORY_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE):
                scores[cat] += 1
    return scores


def classify_regex(prompt: str) -> Tuple[str, int]:
    """Score every category and return ``(best_category, best_score)``.
    Kept for the fallback path and for unit tests that call it directly."""
    scores = _score_categories(prompt)
    best_cat, best_score = FACTUAL_KNOWLEDGE, 0
    for cat in PRIORITY:
        if scores[cat] > best_score:
            best_cat, best_score = cat, scores[cat]
    return best_cat, best_score


def classify(prompt: str, local_model=None) -> str:
    """
    Hybrid classifier.

    Regex is trusted first when it has a real, specific signal (score >= 1) —
    those patterns (```, "bug", "sentiment", "summarize", clue-style logic
    phrasing, etc.) are precise enough that a hit is rarely wrong. The local
    model is used only to fill in when regex found NOTHING (score == 0),
    since that's the only case where we actually need it to disambiguate.

    We do NOT trust the local model when it says "factual_knowledge" over
    regex's own top guess (even a weak one), because factual_knowledge is
    both the model's default fallback label AND the first option listed in
    its classification prompt -- it's the label a small greedy-decoded model
    reaches for when it isn't sure, which is exactly the case we can't trust.
    """
    from categories.local_model import LocalModelSingleton, classify_with_local_model

    scores = _score_categories(prompt)
    regex_cat, regex_score = FACTUAL_KNOWLEDGE, 0
    for cat in PRIORITY:
        if scores[cat] > regex_score:
            regex_cat, regex_score = cat, scores[cat]

    # Regex found a real signal -- trust it immediately, no local model call.
    if regex_score >= 1:
        return _apply_ner_guard(regex_cat, prompt, scores)

    # Regex found nothing at all -- ask the local model to disambiguate.
    loaded_model: Optional[LocalModelSingleton] = (
        local_model if (local_model is not None and local_model.is_loaded()) else None
    )
    if loaded_model is not None:
        lm_cat = classify_with_local_model(prompt, loaded_model)
        # Reject a local-model "factual_knowledge" verdict here specifically --
        # if regex ALSO found nothing, this is the exact ambiguous case where
        # the model's order-bias default is least trustworthy. Only accept
        # a non-default category from it.
        if lm_cat is not None and lm_cat != FACTUAL_KNOWLEDGE:
            return _apply_ner_guard(lm_cat, prompt, scores)

    return FACTUAL_KNOWLEDGE

def _apply_ner_guard(category: str, prompt: str, scores: dict) -> str:
    """
    If the chosen category is ``ner`` but the prompt has no explicit NER
    framing, reject it and fall back to the next-best scoring category
    (excluding ner), or ``factual_knowledge`` if nothing else scored.
    """
    if category != NER:
        return category
    if _ner_is_plausible(prompt):
        return NER

    # Find next-best category (excluding ner).
    for cat in PRIORITY:
        if cat != NER and scores.get(cat, 0) > 0:
            return cat
    return FACTUAL_KNOWLEDGE

