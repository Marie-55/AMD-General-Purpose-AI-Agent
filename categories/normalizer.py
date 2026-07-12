"""
Prompt normalizer.

We don't know what phrasing/style the hidden eval prompts will use, so this
layer:
  1. Strips common conversational fluff/preambles via regex (cheap, no model
     call) — the pattern list is deliberately broad to catch many variants.
  2. Wraps the core instruction with a short, category-specific system message
     that tells the model exactly what shape of answer to produce and to skip
     unnecessary verbosity — this is what actually controls token spend, since
     completion tokens are the lever we control per response.

This is intentionally NOT an LLM-based rewrite: an extra model call to
"clean up" the prompt would cost tokens and latency for every single task,
which is the opposite of what we want.  Regex-level normalization is free
and fast enough to run inline.

Note: prompt compression (TF-IDF sentence scoring) is applied *after*
normalization by the caller (main.py) via compress_messages(), not here,
so this module stays stateless and side-effect-free.
"""
import re

from config import CATEGORY_MAX_TOKENS, DEFAULT_MAX_TOKENS
from categories.prompts import CATEGORY_INSTRUCTIONS, DEFAULT_INSTRUCTION

# ---------------------------------------------------------------------------
# Fluff / preamble patterns (applied in order, case-insensitive, multiline)
# ---------------------------------------------------------------------------
# Each pattern matches a leading phrase that adds no information. We strip
# these before wrapping with the system instruction so the model only sees
# the substance of the request.
FLUFF_PATTERNS = [
    # Factual / general
    r"^quick question\s*[-–—:]*\s*",
    r"^honestly[,.]?\s*",
    r"^give a concise explanation\.?\s*",
    r"^please\s+",
    r"^can you\s+",
    r"^could you\s+",
    r"^i (want|need|would like) (you to\s+)?",

    # Math / reasoning
    r"^solve the problem:\s*",
    r"^solve this:\s*",
    r"^work through the problem:\s*",
    r"^solve step by step:\s*",
    r"^calculate\s+(the\s+)?following:\s*",

    # Logic puzzles
    r"^here is a constraint puzzle\s*[-–—:]*\s*work out the unique solution:\s*",
    r"^here is a (constraint |logic )?puzzle\.?\s*",
    r"^solve this logic puzzle\.?\s*(make sure every clue is satisfied:)?\s*",
    r"^here is a logic puzzle\.?\s*",
    r"^logic puzzle:\s*",

    # Summarization
    r"^please summarize (the following|this)[\s:]*",
    r"^summarize (the following|this)[\s:]*",
    r"^provide a summary of[\s:]*",

    # NER
    r"^please extract (all )?named entities (from|in)[\s:]*",
    r"^extract (all )?named entities (from|in)[\s:]*",
    r"^identify (and label )?every entity[\s:]*",
    r"^list the named entities found here[,.]?\s*",

    # Code
    r"^the following function has a bug\.?\s*",
    r"^here is some code with a bug\.?\s*",
    r"^fix the following (function|code):\s*",
    r"^write a (short\s+)?python (function|script|program) (that|to|which)\s+",
    r"^implement (the following|this) in python:\s*",
    r"^implement this:\s*",

    # Generic trailing noise
    r"\s*please\s+respond\s+concisely\.?\s*$",
    r"\s*be\s+concise\.?\s*$",
]

# Compile once at module load for performance.
_FLUFF_RE = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in FLUFF_PATTERNS]


def get_max_tokens(category: str) -> int:
    """Return the Fireworks max_tokens budget for a given category."""
    return CATEGORY_MAX_TOKENS.get(category, DEFAULT_MAX_TOKENS)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _strip_fluff(text: str) -> str:
    """Apply all fluff-stripping patterns in sequence and return cleaned text."""
    for pat in _FLUFF_RE:
        text = pat.sub("", text)
    # Collapse triple+ blank lines left behind by stripping.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Summarization post-processor
# ---------------------------------------------------------------------------
# Deliberately does NOT trim/cut the answer to force an exact word, sentence,
# or bullet count, even when the request asked for one and the model's
# answer runs over. Mechanically chopping a coherent answer mid-thought to
# hit a count produces worse output for an LLM judge than a coherent answer
# that's merely longer than requested -- compliance with exact-count
# requests is handled entirely by the system prompt (see prompts.py's
# SUMMARIZATION / SUMMARIZATION_LOCAL_SYSTEM instructions), never by
# post-hoc surgery on the model's output.


def enforce_summarization_constraint(answer: str, prompt: str) -> str:
    """Pass the summarization answer through unchanged (aside from
    whitespace trimming). Kept as a named hook so the prompt-compliance
    strategy can be revisited in one place if needed, without callers
    caring whether any actual post-processing happens."""
    return (answer or "").strip()


def normalize_prompt(raw_prompt: str, category: str) -> list:
    """Return a chat 'messages' list ready to send to the model.

    Parameters
    ----------
    raw_prompt:
        The original user-facing prompt text.
    category:
        One of the 8 fixed category strings from the classifier.

    Returns
    -------
    list
        A two-element list: [system_message_dict, user_message_dict], where
        each dict has 'role' and 'content' keys (OpenAI chat format).
    """
    text = _strip_fluff(raw_prompt)

    instruction = CATEGORY_INSTRUCTIONS.get(category, DEFAULT_INSTRUCTION)
    system_msg = f"You are a precise task-solving assistant. {instruction}"

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": text},
    ]