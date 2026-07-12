"""
Single source of truth for every literal prompt string sent to a model
(local GGUF or Fireworks).

Before this module existed, the same prompt idea was frequently defined
independently in 2-3 places (e.g. three different sentiment system prompts
across main.py and local_model.py) with no indication they were meant to
stay in sync. This module does not yet unify those near-duplicates -- that
is a deliberate follow-up decision, not an oversight, because merging them
silently would change which sentiment/summarization labels each path
accepts. What it does do is make every prompt readable and editable in one
place instead of hidden inside business logic.

Organized by the category/module that originally owned each string.
"""
from categories.task_categories import (
    FACTUAL_KNOWLEDGE,
    MATH_REASONING,
    SENTIMENT,
    SUMMARIZATION,
    NER,
    CODE_DEBUGGING,
    LOGIC_PUZZLE,
    CODE_GENERATION,
)

# ---------------------------------------------------------------------------
# Local-model (categories/local_model.py) prompts
# ---------------------------------------------------------------------------

CLASSIFIER_SYSTEM = (
    "You are a task classifier. Classify the user's request into exactly one of these "
    "categories: factual_knowledge, math_reasoning, sentiment, summarization, ner, "
    "code_debugging, logic_puzzle, code_generation. "
    "Reply with ONLY the category name, nothing else."
)

SENTIMENT_LOCAL_SYSTEM = (
    "Classify the sentiment of the text as positive, negative, mixed, or neutral. "
    "Use 'mixed' when the text contains both clear positive and negative elements. "
    "Reply with exactly ONE sentence. "
    "Start with the label (positive, negative, mixed, or neutral), then a colon, "
    "then a brief justification. No bullets, no extra sentences, no markdown."
)

FACTUAL_LOCAL_SYSTEM = (
    "Answer the user's question directly and completely. Cover every distinct part of the "
    "question -- if it asks to compare or distinguish two things, address both explicitly, "
    "not just one. Use as few sentences as that requires (often 2-4, more if the question has "
    "several distinct parts); do not omit a required point just to save length. "
    "No preamble, no restating the question, no unrelated background, and no markdown unless "
    "it is clearly helpful."
)

SUMMARIZATION_LOCAL_SYSTEM = (
    "Summarize the text as concisely and clearly as possible. "
    "Always write complete sentences — never stop mid-sentence. "
    "If the request specifies an exact number of sentences or bullet points, or a maximum word "
    "count per bullet, follow it exactly — that constraint is graded directly. "
    "Preserve all key facts. Do not add preambles or extra commentary. "
    "Never apologize or refuse."
)

LOGIC_LOCAL_SYSTEM = (
    "You are a logical reasoning expert. "
    "Work through the clues one by one, eliminating options as you go. "
    "End your answer with a single line starting with 'Answer:' followed "
    "by the complete solution in plain language. "
    "No code, no markdown, no bullet points — just clear reasoning and the answer."
)

NER_LOCAL_SYSTEM = (
    "You are a named entity recognition system. "
    "Extract all named entities from the text. "
    "The request may specify which entity types to use and exactly how to label them -- if it "
    "does, follow those instructions precisely and use its exact type labels. "
    "If the request does not specify entity types, use standard types such as PERSON, "
    "ORGANIZATION, LOCATION, and DATE. "
    "Return ONLY a JSON array. Each element must be an object with exactly "
    "two keys: 'text' (the entity string) and 'type' (the entity type, per the rule above). "
    "Do NOT include the entire sentence as an entity. "
    "Example output: "
    '[{"text": "Paris", "type": "LOCATION"}, {"text": "2024", "type": "DATE"}]. '
    "Do not include any explanation, markdown fences, or keys other than 'text' and 'type'."
)

NER_LOCAL_RETRY_SYSTEM = (
    "You are a named entity recognizer. Return ONLY a valid JSON array of objects with keys text and type. "
    "Use the entity types the request specifies, with its exact labels; if it specifies none, use "
    "standard types such as PERSON, ORGANIZATION, LOCATION, DATE. "
    "Do not add titles, prose, markdown, or explanations. "
    "If there are no entities, return []."
)

MATH_CODE_LOCAL_SYSTEM = (
    "You are a Python code generator. Solve the math problem by writing a complete Python script.\n"
    "Your output must be a single code block starting with ```python and ending with ```.\n"
    "Do not include any explanation, comments, or extra text outside the code block.\n"
    "Print each intermediate calculation on its own line as you compute it, so the work is visible "
    "in the printed output (e.g. print the running value after each step).\n"
    "IMPORTANT: The LAST line of your code MUST be a print() statement that outputs the final answer, "
    "and it must be the last thing printed.\n"
    "Example:\n"
    "```python\n"
    "x = 5\n"
    "y = 10\n"
    "print(f\"x + y = {x + y}\")\n"
    "RESULT = x + y\n"
    "print(RESULT)\n"
    "```\n"
    "Another example:\n"
    "```python\n"
    "price = 160 * 0.75\n"
    "print(f\"price after discount: {price}\")\n"
    "total = price * 1.08\n"
    "print(total)\n"
    "```\n"
    "Now solve this problem:\n"
)

MATH_EXPLANATION_LOCAL_SYSTEM = (
    "You are a math reasoning expert. Work through the problem step by step, "
    "showing each calculation clearly. End with a line starting with 'Answer:' "
    "followed by the final numeric value. Keep your response concise but complete."
)

# ---------------------------------------------------------------------------
# categories/code_exec.py prompts
# ---------------------------------------------------------------------------

CODE_GEN_SYSTEM = (
    "You are a Python code generator. Given a math problem, write a SELF-CONTAINED "
    "Python script using only the standard library (no external packages, no input()). "
    "Use the fractions.Fraction class for any division or fraction arithmetic to avoid "
    "floating-point errors. "
    "Print each intermediate calculation on its own line as you compute it (e.g. "
    "print(f\"after discount: {value}\")), so the work is visible in the printed output. "
    "Assign the final answer to a variable named RESULT. "
    "If RESULT is a float or Fraction, convert to float and round to 2 decimal places: "
    "RESULT = round(float(RESULT), 2). "
    "If the result is a whole number after rounding (e.g. 600.0), print it as an integer: "
    "print(int(RESULT) if RESULT == int(RESULT) else RESULT). "
    "This RESULT print must be the LAST line printed. "
    "Return only one complete ```python code block and nothing else. "
    "No explanation, no markdown outside the block, and no extra text."
)

# ---------------------------------------------------------------------------
# categories/code_verifier.py prompts
# ---------------------------------------------------------------------------

CODE_VERIFY_FALLBACK_SYSTEM: dict[str, str] = {
    CODE_GENERATION: (
        "You are a Python programming expert. The previous code attempt could not be "
        "verified. Answer the user's request as clearly as possible in natural language, "
        "and if you include code wrap it in a single ```python block."
    ),
    CODE_DEBUGGING: (
        "You are a Python debugging expert. The previous code attempt could not be "
        "verified. Return valid JSON only with keys issues and corrected_parts. "
        "issues must be a short array of bug descriptions. corrected_parts must be an "
        "array of objects containing only the corrected code snippets and optional "
        "location/original fields. If you include code, put it in corrected_parts."
    ),
}

CODE_VERIFY_DEFAULT_FALLBACK_SYSTEM = (
    "You are a helpful assistant. Answer the user's request as clearly and concisely "
    "as possible."
)

CODE_VERIFY_FIX_SYSTEM = (
    "You are a Python expert. Fix the following code so it is syntactically correct "
    "and runs without errors. Return valid JSON only. For code_generation include a "
    "corrected_code field containing the full Python code. For code_debugging include "
    "issues and corrected_parts fields, where corrected_parts contains the corrected "
    "code snippets."
)

# ---------------------------------------------------------------------------
# categories/normalizer.py prompts
# ---------------------------------------------------------------------------

CATEGORY_INSTRUCTIONS: dict[str, str] = {
    FACTUAL_KNOWLEDGE: (
        "Answer directly and completely. Cover every distinct part of the question -- if it asks "
        "to compare or distinguish two things, address both explicitly, not just one. Use as few "
        "sentences as that requires; do not omit a required point just to save length. "
        "Do not provide unrelated background context. Maximize clarity per token, but never at "
        "the cost of a missing point."
    ),
    MATH_REASONING: (
        "Show brief step-by-step arithmetic, then finish with a single line: "
        "'Answer: <value>'."
    ),
    SENTIMENT: (
        "Classify the sentiment and reply with exactly ONE sentence. "
        "Start the sentence with the label (positive, negative, or mixed, or neutral), "
        "then briefly justify it in the same sentence. "
        "No bullets, no extra sentences, no markdown."
    ),
    SUMMARIZATION: (
        "Answer as concisely and clearly as possible. "
        "Do not reason step by step in your response — output only the final summary, nothing else. "
        "If the request specifies an exact number of sentences or bullet points, or a maximum word "
        "count per bullet, follow it exactly: that constraint is graded directly, and producing more "
        "or fewer sentences/bullets than requested fails regardless of content quality. "
        "Minimize output tokens while preserving the key meaning. "
        "Do not add preambles or extra commentary. "
        "Never apologize, refuse, or state that you cannot determine an answer."
    ),
    NER: (
        "Extract ALL named entities as a JSON array of objects with keys 'text' and 'type'. "
        "The request may specify which entity types to use and its exact labels for them -- if it "
        "does, follow that specification precisely. If it does not, use standard types such as "
        "PERSON, ORGANIZATION, LOCATION, and DATE. "
        "Each 'text' must be a short, standalone name or date, not a whole sentence. "
        "Example: [{\"text\": \"Marie Curie\", \"type\": \"PERSON\"}, {\"text\": \"Google\", \"type\": \"ORGANIZATION\"}]. "
        "Return ONLY the JSON array, no other text."
    ),
    CODE_DEBUGGING: (
        "Explain the bug(s) in plain English, then provide the corrected code. "
        "Format your response as clean markdown: "
        "start with a '## Bug Explanation' section describing the issue(s), "
        "then a '## Corrected Code' section containing the full fixed code in a ```python block. "
        "Do not wrap your response in JSON. Do not add unnecessary preamble."
    ),
    LOGIC_PUZZLE: (
        "Work through the clues step by step in your response. "
        "After your reasoning, end with a single line starting exactly with 'Answer:' "
        "followed by the complete concrete solution in plain language. "
        "No code, no tables, no markdown headers. Keep your reasoning concise."
    ),
    CODE_GENERATION: (
        "Return ONLY a Python code block starting with ```python and ending with ```. "
        "Do not include any explanation, comments, or text outside the code block. "
        "The code must be complete and runnable."
    ),
}

DEFAULT_INSTRUCTION = "Answer clearly and concisely. Avoid unnecessary preamble."

# ---------------------------------------------------------------------------
# main.py inline prompts (per-category Fireworks escalation / retry paths)
# ---------------------------------------------------------------------------

SENTIMENT_FIREWORKS_SYSTEM = (
    "Classify the sentiment of the text. "
    "Reply with exactly one sentence. "
    "Start with the label — positive, negative, or neutral — "
    "then a colon, then a brief one-clause justification. "
    "No bullets, no extra sentences, no markdown."
)

SENTIMENT_FIREWORKS_STRICT_SYSTEM = (
    "Classify the sentiment. "
    "Output exactly one sentence. "
    "The sentence MUST start with one of: positive, negative, neutral — "
    "then a colon and a short explanation. "
    "No other words before the label. No reasoning. No markdown."
)

SUMMARIZATION_FIREWORKS_STRICT_SYSTEM = (
    "You are a summarization assistant. Output only the summary. "
    "Do not include any reasoning, explanation, or additional text. "
    "Start your response with the summary immediately."
)

SUMMARIZATION_FIREWORKS_USER_TEMPLATE = "Summary of the following text:\n\n{prompt}\n\nSummary:"

SUMMARIZATION_FIREWORKS_RETRY_SYSTEM = "Summarize the text concisely. Return only the summary."

CODE_GENERATION_INCOMPLETE_RETRY_SYSTEM = (
    "You are a Python code generator. Return a complete, runnable Python function that "
    "solves the problem. Do not include explanations. The code must have a non‑empty "
    "function body."
)

CODE_BLOCK_MISSING_RETRY_PREFIX = (
    "Return ONLY a Python code block starting with ```python and ending with ```. "
    "No text before or after. Implement: "
)

LOGIC_TERSE_RETRY_SYSTEM = (
    "Solve this logic puzzle. "
    "Show brief reasoning, then end with 'Answer:' and the solution. "
    "Be direct and concise."
)

MATH_TERSE_SYSTEM = (
    "Solve the problem. Work it out, then redo the calculation once more independently to "
    "check it before answering. Reply with ONLY the final numeric value, nothing else."
)

# Used only as the final math fallback, when both the local and Fireworks
# code-generation attempts have already failed and there is no
# code-execution result left to verify the answer against. This is the one
# math path where an arithmetic mistake can reach the final answer
# undetected, so it explicitly asks for a self-check pass rather than using
# the generic math instruction.
MATH_NL_FALLBACK_VERIFIED_SYSTEM = (
    "Solve this math problem step by step, showing your arithmetic clearly. "
    "After computing the final answer, redo the calculation once more independently "
    "to check it. If the two attempts disagree, work out which one is correct before "
    "answering. Finish with a single line: 'Answer: <value>'."
)

# Per-category suffix appended to the normalized system prompt when a first
# attempt came back empty (see main.py:_recover_empty_answer).
RECOVER_EMPTY_ANSWER_EXTRA: dict[str, str] = {
    FACTUAL_KNOWLEDGE: " Answer directly and do not return an empty response.",
    MATH_REASONING: " Provide the final numeric answer even if the reasoning is brief.",
    SENTIMENT: " Reply with one sentence: start with the label (positive, negative, or mixed) then briefly justify it.",
    SUMMARIZATION: " Return a non-empty summary that obeys the requested format.",
    NER: " Return a valid JSON array. If there are no entities, return [].",
    CODE_DEBUGGING: " Return a markdown explanation with a Bug Explanation section and a Corrected Code section.",
    LOGIC_PUZZLE: " State one valid final answer clearly.",
    CODE_GENERATION: " Return a complete Python code block only.",
}
RECOVER_EMPTY_ANSWER_DEFAULT_EXTRA = " Return a non-empty answer."
