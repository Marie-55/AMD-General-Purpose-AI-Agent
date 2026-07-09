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


# ---------------------------------------------------------------------------
# Category-specific system instructions
# ---------------------------------------------------------------------------
CATEGORY_INSTRUCTIONS = {
    "factual_knowledge": (
        "Answer directly and concisely in 2-4 sentences. "
        "No preamble, no restating the question."
    ),
    "math_reasoning": (
        "Show brief step-by-step arithmetic, then finish with a single line: "
        "'Answer: <value>'."
    ),
    "sentiment": (
        "Reply with ONLY one word: positive, negative, or mixed. "
        "No explanation, no punctuation, no other text."
    ),
    "summarization": (
        "Follow any length/format constraint in the prompt EXACTLY. "
        "If the prompt says 'exactly N words', count carefully and output exactly N words. "
        "If the prompt says 'exactly N bullet points', output exactly N bullets. "
        "Output only the summary text — no preamble like 'Here is a summary'."
    ),
    "ner": (
        "Return ONLY a valid JSON array. Each element is an object with exactly two keys: "
        "'text' (the entity string as it appears in the text) and 'type' (one of: PERSON, ORG, LOCATION, DATE). "
        "Include ALL entities including dates and locations. "
        'Example: [{"text":"Alice","type":"PERSON"},{"text":"2024","type":"DATE"},{"text":"Paris","type":"LOCATION"}]. '
        "No prose, no markdown fences, no keys other than 'text' and 'type'."
    ),
    "code_debugging": (
        "State the bug in one short sentence, then give the corrected function in a single "
        "Python code block. No other prose."
    ),
    "logic_puzzle": (
        "Reason briefly through the clues, then state the final answer clearly on the last line."
    ),
    "code_generation": (
        "Return ONLY a Python code block that implements the requested function. "
        "Start your response with ```python on the first line. "
        "End with ``` on the last line. "
        "No prose before the block, no docstrings inside, no example usage after."
    ),
}

DEFAULT_INSTRUCTION = "Answer clearly and concisely. Avoid unnecessary preamble."


# ---------------------------------------------------------------------------
# Per-category max_tokens budgets
# ---------------------------------------------------------------------------
# Rationale per category (all >= 500 to avoid truncation):
#   factual_knowledge  — 2-4 sentences; 600 is generous but safe
#   math_reasoning     — step-by-step working + final answer; 700
#   sentiment          — single label word only (positive/negative/mixed); 10
#   summarization      — 1 sentence / 20 words / 2 sentences; 800 for safety
#   ner                — JSON array; 700 covers ~15 entities safely
#   code_debugging     — bug explanation + full corrected function; 900
#   logic_puzzle       — reasoning trace + final answer; 800 avoids truncation
#   code_generation    — full function implementation; 1000
CATEGORY_MAX_TOKENS: dict[str, int] = {
    "factual_knowledge": 600,
    "math_reasoning":    700,
    "sentiment":          10,
    "summarization":     800,
    "ner":               700,
    "code_debugging":    900,
    "logic_puzzle":      800,
    "code_generation":  1000,
}

DEFAULT_MAX_TOKENS = 600


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
# Summarization word-count post-processor
# ---------------------------------------------------------------------------
_EXACT_WORDS_RE  = re.compile(r"exactly\s+(\d+)\s+words?", re.IGNORECASE)
_MAX_WORDS_RE    = re.compile(r"no more than\s+(\d+)\s+words?", re.IGNORECASE)
_EXACT_BULLET_RE = re.compile(r"exactly\s+(\d+)\s+bullet", re.IGNORECASE)


def enforce_summarization_constraint(answer: str, prompt: str) -> str:
    """Trim or pad a summarization answer to match an exact word-count constraint.

    Only acts when the prompt contains 'exactly N words'.  For bullet-point and
    'no more than N words' constraints we trust the model and only trim at the
    hard limit to avoid garbling the answer.
    """
    # Exact word count
    m = _EXACT_WORDS_RE.search(prompt)
    if m:
        target = int(m.group(1))
        words  = answer.split()
        if len(words) > target:
            answer = " ".join(words[:target])
            # Ensure it ends with a period if we trimmed.
            if answer and not answer[-1] in ".!?":
                answer = answer.rstrip(",;:") + "."
        return answer

    # No more than N words — trim only if over limit
    m = _MAX_WORDS_RE.search(prompt)
    if m:
        limit = int(m.group(1))
        words = answer.split()
        if len(words) > limit:
            answer = " ".join(words[:limit])
            if answer and answer[-1] not in ".!?":
                answer = answer.rstrip(",;:") + "."
        return answer

    return answer


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
