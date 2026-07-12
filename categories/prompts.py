"""Every system prompt used by the pipeline, one per category, plus the
two variants used by the math self-consistency check and the summarization
retry. Keeping these here (instead of inline in handlers.py) means a prompt
tweak never requires touching pipeline logic.
"""

FACTUAL_SYSTEM = (
    "You are a knowledgeable assistant. Answer the question directly and "
    "completely. Be concise but do not omit any part of what the question asks for."
)

SENTIMENT_SYSTEM = (
    "Classify the sentiment of the given text as Positive, Negative, Neutral, "
    "or Mixed. If the text expresses both positive and negative points, "
    "acknowledge both sides rather than picking one -- do not default to "
    "Negative just because a complaint is present. Give the label followed "
    "by a one-sentence justification."
)

# Never mechanically trim a summary to hit a word/sentence/bullet count --
# compliance comes entirely from prompt-following, not post-processing.
SUMMARIZATION_SYSTEM = (
    "Summarize the given text, following the exact constraint stated in the "
    "request (word count, sentence count, bullet count, etc.) as closely as "
    "possible. Write only the summary itself, with no preamble or labels."
)
SUMMARIZATION_RETRY_SYSTEM = (
    "Summarize the given text as instructed. Reply with ONLY the summary, "
    "nothing else, and keep it short and complete -- do not leave a "
    "sentence unfinished."
)

# Entity types are NOT hardcoded here: if the request specifies its own
# types, those win. This prompt only supplies a default for when it doesn't.
NER_SYSTEM = (
    "Extract all named entities from the text. If the request specifies which entity types to use, follow those instructions exactly and use its exact type labels. Otherwise use standard types such as PERSON, ORGANIZATION, LOCATION, and DATE. Return a JSON array where each element has a 'text' key (the entity string) and a 'type' key. Do not include the entire sentence as a single entity."
)

LOGIC_SYSTEM = (
    "Solve the logic puzzle step by step, applying each clue in turn and "
    "eliminating impossible options. End your response with a single line: "
    "'Answer: <final answer>'."
)

# Math is solved primarily via code (see math_solver.py) since execution is
# checkable; this is the codegen half of that check.
MATH_CODE_SYSTEM = (
    "Write a short Python script that solves the given math word problem. "
    "Compute the answer using code, not by writing the final number "
    "directly, and end the script with a single print() statement that "
    "outputs only the final numeric answer. Return ONLY a Python code "
    "block, no explanation."
)

# The independent natural-language cross-check: code that runs successfully
# can still encode the wrong arithmetic, so this asks for a fully separate
# derivation to compare against.
MATH_NL_SYSTEM = (
    "Solve this math problem step by step, showing your arithmetic clearly. "
    "After computing the final answer, redo the calculation once more "
    "independently to check it; if the two attempts disagree, work out "
    "which one is correct before answering. Finish with a single line: "
    "'Answer: <value>'."
)

CODE_DEBUG_SYSTEM = (
    "You are given code with a bug. Identify the bug, explain it briefly, "
    "and provide the corrected code in a Python code block."
)

CODE_GEN_SYSTEM = (
    "Write clean, correct Python code that satisfies the request. Return "
    "the code in a single Python code block. Keep explanation minimal -- "
    "prioritize a complete, runnable implementation over commentary."
)

# Used only when the regex classifier's own signal is weak (see
# categories/classifier.py) -- asks the model to name exactly one of the 8
# category labels, nothing else, so the reply can be matched by substring.
CLASSIFY_SYSTEM = (
    "Classify the following request into exactly one category. Reply with "
    "ONLY the category name, nothing else.\n\n"
    "Categories:\n"
    "- factual_knowledge: explaining a concept, definition, or how/why something works\n"
    "- math_reasoning: arithmetic, percentages, word problems with numbers\n"
    "- sentiment: classifying the sentiment/tone of a piece of text\n"
    "- summarization: condensing a passage under a length constraint\n"
    "- ner: extracting named entities (people, organizations, places, dates, etc.)\n"
    "- code_debugging: given code with a bug, find and fix it\n"
    "- logic_puzzle: deduce an answer from a set of constraints/clues\n"
    "- code_generation: write new code from a specification"
)
