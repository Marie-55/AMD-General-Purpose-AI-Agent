"""
Prompt normalizer.

We don't know what phrasing/style the hidden eval prompts will use ("Quick
question --", "Work through the problem:", "Here is a constraint puzzle --",
etc.), so this layer:
  1. Strips common conversational fluff/preambles via regex (cheap, no model call).
  2. Wraps the core instruction with a short, category-specific system message
     that tells the model exactly what shape of answer to produce and to skip
     unnecessary verbosity -- this is what actually controls token spend, since
     completion tokens are the lever we control per response.

This is intentionally NOT an LLM-based rewrite: an extra model call to
"clean up" the prompt would cost tokens and latency for every single task,
which is the opposite of what we want. Regex-level normalization is free and
fast enough to run inline.
"""
import re

FLUFF_PATTERNS = [
    r"^quick question\s*--\s*",
    r"^here is a constraint puzzle\s*--\s*work out the unique solution:\s*",
    r"^here is a logic puzzle\.?\s*",
    r"^work through the problem:\s*",
    r"^solve this logic puzzle\.\s*make sure every clue is satisfied:\s*",
    r"^solve the problem:\s*",
    r"^give a concise explanation\.?\s*",
    r"^honestly,?\s*",
]

CATEGORY_INSTRUCTIONS = {
    "factual_knowledge": (
        "Answer directly and concisely in 2-4 sentences. No preamble, no restating the question."
    ),
    "math_reasoning": (
        "Show brief step-by-step arithmetic, then finish with a single line: 'Answer: <value>'."
    ),
    "sentiment": (
        "Reply with the sentiment label (positive/negative/mixed) followed by one short "
        "justifying sentence. Nothing else."
    ),
    "summarization": (
        "Follow any length/format constraint in the prompt exactly. Output only the summary text, "
        "no preamble like 'Here is a summary'."
    ),
    "ner": (
        "Return ONLY valid JSON: a list of objects each with 'text' and 'type' "
        "(PERSON/ORG/LOCATION/DATE). No extra prose, no markdown fences."
    ),
    "code_debugging": (
        "State the bug in one short sentence, then give the corrected function in a single "
        "Python code block. No other prose."
    ),
    "logic_puzzle": (
        "Reason briefly through the clues, then state the final answer clearly on the last line."
    ),
    "code_generation": (
        "Return ONLY a single Python code block implementing the requested function. "
        "No explanation before or after."
    ),
}

DEFAULT_INSTRUCTION = "Answer clearly and concisely. Avoid unnecessary preamble."


def normalize_prompt(raw_prompt: str, category: str) -> list:
    """Return a chat 'messages' list ready to send to the model."""
    text = raw_prompt.strip()
    for pat in FLUFF_PATTERNS:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    instruction = CATEGORY_INSTRUCTIONS.get(category, DEFAULT_INSTRUCTION)
    system_msg = f"You are a precise task-solving assistant. {instruction}"
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": text},
    ]