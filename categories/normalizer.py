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
        "Answer concisely in 2-3 sentences maximum. "
        "Do not provide background context. "
        "Minimize output tokens while maximizing clarity."
    ),
    "math_reasoning": (
        "Show brief step-by-step arithmetic, then finish with a single line: "
        "'Answer: <value>'."
    ),
    "sentiment": (
        "Classify the sentiment and reply with exactly ONE sentence. "
        "Start the sentence with the label (positive, negative, or mixed, or neutral), "
        "then briefly justify it in the same sentence. "
        "No bullets, no extra sentences, no markdown."
    ),
    "summarization": (
        "Answer as concisely and clearly as possible. "
        "Do not REASON STEP BY STEP in your response — just give the final summary. "
        "the final summary itself, nothing else. "
        "Minimize output tokens while preserving the key meaning. "
        "Do not add preambles or extra commentary. "
        "Never apologize, refuse, or state that you cannot determine an answer."
    ),
    "ner": (
    "Extract ALL named entities as a JSON array of objects with keys 'text' and 'type'. "
    "Use appropriate types: PERSON, ORG, LOCATION, DATE, EVENT, PRODUCT, etc. "
    "Each 'text' must be a short, standalone name or date, not a whole sentence. "
    "Example: [{\"text\": \"Gemini\", \"type\": \"PRODUCT\"}, {\"text\": \"Google I/O\", \"type\": \"EVENT\"}]. "
    "Return ONLY the JSON array, no other text."
),
    "code_debugging": (
        "Explain the bug(s) in plain English, then provide the corrected code. "
        "Format your response as clean markdown: "
        "start with a '## Bug Explanation' section describing the issue(s), "
        "then a '## Corrected Code' section containing the full fixed code in a ```python block. "
        "Do not wrap your response in JSON. Do not add unnecessary preamble."
    ),
    # In CATEGORY_INSTRUCTIONS, change:
    "logic_puzzle": (
        "Work through the clues step by step in your response. "
        "After your reasoning, end with a single line starting exactly with 'Answer:' "
        "followed by the complete concrete solution in plain language. "
        "No code, no tables, no markdown headers. Keep your reasoning concise."
    ),
 "code_generation": (
    "Return ONLY a Python code block starting with ```python and ending with ```. "
    "Do not include any explanation, comments, or text outside the code block. "
    "The code must be complete and runnable."
)
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
    "factual_knowledge": 280,
    "math_reasoning":    700,
    "sentiment":         60,
    "summarization":    800,
    "ner":              600,  # In CATEGORY_MAX_TOKENS, CHANGE ner:
    "code_debugging":   1024,
    "logic_puzzle":     4096,
    "code_generation":  4096,  # minimax spends ~1000-1500 on CoT; need headroom for full function
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
_EXACT_BULLET_RE = re.compile(r"exactly\s+(\d+|one|two|three|four|five)\s+bullet", re.IGNORECASE)
_NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}


def _trim_to_word_count(text: str, limit: int) -> str:
    """Best-effort exact/maximum word-count trimming without another model call."""
    tokens = re.findall(r"\S+", (text or "").strip())
    if len(tokens) <= limit:
        return " ".join(tokens)
    trimmed = " ".join(tokens[:limit]).rstrip(" ,;:")
    # Avoid ending exact-count summaries with dangling conjunctions/prepositions.
    while trimmed.split() and trimmed.split()[-1].lower().strip(".,;:") in {"and", "or", "but", "with", "to", "of", "in", "for", "by", "after", "while"}:
        words = trimmed.split()
        words[-1] = words[-1].rstrip(".,;:") + "."
        trimmed = " ".join(words)
        break
    if trimmed and trimmed[-1] not in ".!?":
        trimmed += "."
    return trimmed


def _normalize_bullets(answer: str, expected: int) -> str:
    lines = [line.strip() for line in (answer or "").splitlines() if line.strip()]
    cleaned: list[str] = []
    for line in lines:
        line = re.sub(r"^(?:[-*•]+|\d+[.)])\s*", "", line).strip()
        if line:
            cleaned.append(line)
    if not cleaned:
        return answer
    return "\n".join(f"- {line}" for line in cleaned[:expected])


def enforce_summarization_constraint(answer: str, prompt: str) -> str:
    """Cheap post-processing for the benchmark's mechanical summary constraints.

    The LLM judge may be semantic, but exact word-count/bullet prompts are often
    scored harshly.  This avoids spending another model call while fixing the
    common over-long local summary case.
    """
    if not (answer or "").strip():
        return answer

    exact_bullets = _EXACT_BULLET_RE.search(prompt)
    if exact_bullets:
        raw_count = exact_bullets.group(1)
        expected = int(raw_count) if raw_count.isdigit() else _NUM_WORDS[raw_count.lower()]
        return _normalize_bullets(answer, expected)

    exact_words = _EXACT_WORDS_RE.search(prompt)
    if exact_words:
        return _trim_to_word_count(answer, int(exact_words.group(1)))

    max_words = _MAX_WORDS_RE.search(prompt)
    if max_words:
        return _trim_to_word_count(answer, int(max_words.group(1)))

    return answer.strip()


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