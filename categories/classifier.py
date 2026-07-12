"""Hybrid task classifier: maps a prompt to one of the 8 competition
categories using regex first, falling back to the generalist model when
the regex signal is weak or absent.

A real run showed the pure-regex classifier silently defaulting to
factual_knowledge for phrasings its patterns didn't anticipate -- e.g.
"Return entities grouped by type as JSON" (an NER task with no "extract"
or "named entit[y]" keyword) and "Give one valid finishing order" (a logic
puzzle with no "determine the order" phrasing). Both got routed through
handle_factual instead of their real handler: the NER task came back as a
malformed JSON object instead of an entity array (no grammar constraint
was ever applied), and the logic task came back as an unverified one-liner
that turned out to violate one of the puzzle's own clues. Regex alone
can't be extended to cover every phrasing, so ambiguous cases are now
resolved by asking the already-loaded generalist model directly.
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


def classify_regex(prompt: str) -> tuple[str, int]:
    """Score every category against *prompt* and return (best_category,
    best_score). score == 0 means no pattern matched anything -- the
    "factual_knowledge" returned in that case is just PRIORITY's fallback
    ordering, not a real signal, and should not be trusted as one.
    """
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
    return best_cat, best_score


def _classify_with_model(prompt: str, runtime) -> str | None:
    from categories import prompts
    import config

    messages = [
        {"role": "system", "content": prompts.CLASSIFY_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    raw = runtime.generate(messages, max_tokens=config.CLASSIFY_MAX_TOKENS, temperature=0.0)
    if not raw:
        return None
    text = raw.lower()
    # Word-boundary match, not a plain substring check: "ner" is a literal
    # substring of "generation" (ge-NER-ation), so a naive `cat in text`
    # matched "ner" inside a correct "code_generation" answer every time --
    # caught in a real run where two clear code_generation prompts were
    # silently routed to the NER handler instead.
    for cat in ALL_CATEGORIES:
        if re.search(rf"\b{re.escape(cat)}\b", text):
            return cat
    return None


def classify(prompt: str, runtime=None) -> str:
    """Regex first; the model only gets a turn when regex's own signal is
    weak (score <= 1) -- a single incidental keyword hit isn't more
    trustworthy than a full-context read from the model, but a strong
    multi-pattern regex match (score >= 2) is specific enough to trust
    outright and skip the extra generation call.

    *runtime* is optional so this stays usable in tests / anywhere a model
    isn't loaded -- classify(prompt) alone still works exactly as before.
    """
    regex_cat, regex_score = classify_regex(prompt)

    if runtime is None or regex_score >= 2:
        return regex_cat

    model_cat = _classify_with_model(prompt, runtime)
    if model_cat is None:
        return regex_cat
    return model_cat
