"""
Static model routing.

IMPORTANT: model IDs are never hardcoded as the literal string sent to the API.
We read ALLOWED_MODELS from the environment at runtime (as required) and pick
the best match for each ROLE using substring keywords. If none of the expected
keywords are found (e.g. the harness publishes different literal IDs on launch
day), we fail safe onto the first allowed model rather than crashing or,
worse, calling something outside ALLOWED_MODELS.
"""
import os


def get_allowed_models():
    raw = os.environ.get("ALLOWED_MODELS", "")
    models = [m.strip() for m in raw.split(",") if m.strip()]
    if not models:
        raise RuntimeError("ALLOWED_MODELS env var is empty or unset -- cannot route any calls.")
    return models


# Keyword hints used only to pick a role among the models actually present in
# ALLOWED_MODELS at runtime -- never used as the literal model string itself.
ROLE_KEYWORDS = {
    "cheap_general": ["gemma-4-26b-a4b-it", "a4b"],
    "cheap_alt": ["gemma-4-31b-it-nvfp4", "nvfp4"],
    "quality_general": ["gemma-4-31b-it"],
    "code_specialist": ["kimi", "code"],
    "reasoning_specialist": ["minimax"],
}

# Which role handles which of the 8 fixed categories.
# Rationale: cheap/short-output categories (facts, sentiment, NER, summarization)
# go to the smallest model that can do them reliably, to minimize tokens.
# Code and logic categories go to the code/reasoning specialists, but ONLY to
# generate a short solving script -- see code_exec.py -- not to reason the
# whole answer out in expensive completion tokens.
CATEGORY_ROLE = {
    "factual_knowledge": "cheap_general",
    "sentiment": "cheap_general",
    "summarization": "cheap_alt",
    "ner": "cheap_general",
    "code_debugging": "code_specialist",
    "code_generation": "code_specialist",
    "math_reasoning": "code_specialist",
    "logic_puzzle": "reasoning_specialist",
}

# Categories where we first try "LLM writes solver code -> we execute it locally"
# before falling back to a direct natural-language answer.
CODE_EXEC_CATEGORIES = {"math_reasoning", "logic_puzzle"}


def _resolve_role(role: str, allowed: list) -> str:
    for kw in ROLE_KEYWORDS.get(role, []):
        for m in allowed:
            if kw in m:
                return m
    return allowed[0]  # safe fallback: first model in the allowed list


def route(category: str) -> str:
    allowed = get_allowed_models()
    role = CATEGORY_ROLE.get(category, "cheap_general")
    return _resolve_role(role, allowed)