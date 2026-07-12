from categories.classifier import classify, classify_regex

# One prompt per category, each phrased to score >=1 on the regex patterns
# so classification is deterministic without a local model.
PROMPTS_BY_CATEGORY = {
    "factual_knowledge": "What causes rainbows to form after rain?",
    "math_reasoning": (
        "A store's price is $80 with a 15% discount and 8% tax. "
        "What is the final price?"
    ),
    "sentiment": (
        "Classify the sentiment of this review: The battery life is amazing "
        "and the camera quality exceeded every expectation."
    ),
    "summarization": (
        "Summarize the following text in one sentence: The quarterly report "
        "shows revenue grew steadily across all regions."
    ),
    "ner": (
        "Extract all named entities from the following text and return them "
        "as a JSON array grouped by type: Barack Obama visited Paris in 2011."
    ),
    "logic_puzzle": (
        "Logic puzzle: Alice, Bob, and Carol sit in a row. Alice is "
        "immediately left of Bob. Determine the seating arrangement."
    ),
    "code_debugging": (
        "The following function has a bug. Find and fix the bug:\n"
        "```python\ndef add(a, b):\n    return a - b\n```"
    ),
    "code_generation": "Write a Python function that returns the factorial of a number.",
}


def test_classify_regex_covers_every_category():
    for expected_category, prompt in PROMPTS_BY_CATEGORY.items():
        category, score = classify_regex(prompt)
        assert category == expected_category, f"{prompt!r} -> {category} (score={score})"
        assert score >= 1


def test_classify_without_local_model_matches_regex():
    for expected_category, prompt in PROMPTS_BY_CATEGORY.items():
        assert classify(prompt, local_model=None) == expected_category


def test_classify_defaults_to_factual_knowledge_on_no_signal():
    # A prompt with no category-specific keywords at all.
    assert classify("banana banana banana", local_model=None) == "factual_knowledge"


def test_ner_guard_rejects_plain_prompts_with_proper_nouns():
    # Contains a person's name but no explicit NER framing -- must NOT be
    # classified as ner (regression guard for the false-positive this module
    # was specifically built to avoid).
    prompt = "Why did Marie Curie win two Nobel Prizes?"
    assert classify(prompt, local_model=None) != "ner"
