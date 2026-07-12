from categories.normalizer import (
    normalize_prompt,
    get_max_tokens,
    enforce_summarization_constraint,
)


def test_normalize_prompt_shape():
    messages = normalize_prompt("Explain quantum entanglement.", "factual_knowledge")
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "distinct part" in messages[0]["content"].lower()
    assert "quantum entanglement" in messages[1]["content"]


def test_normalize_prompt_strips_fluff_preamble():
    messages = normalize_prompt("Please summarize the following: Long report text.", "summarization")
    assert messages[1]["content"] == "Long report text."


def test_normalize_prompt_unknown_category_uses_default_instruction():
    messages = normalize_prompt("hello", "not_a_real_category")
    assert "Answer clearly and concisely" in messages[0]["content"]


def test_get_max_tokens_known_and_default():
    assert get_max_tokens("sentiment") == 220
    assert get_max_tokens("totally_unknown_category") == 600


def test_enforce_summarization_constraint_noop_when_no_constraint():
    answer = "A plain summary sentence."
    prompt = "Summarize this text."
    assert enforce_summarization_constraint(answer, prompt) == answer


def test_enforce_summarization_constraint_never_trims_over_word_count():
    answer = "one two three four five six seven eight nine ten"
    prompt = "Summarize in exactly 5 words."
    result = enforce_summarization_constraint(answer, prompt)
    assert result == answer
    assert len(result.split()) == 10


def test_enforce_summarization_constraint_never_trims_over_sentence_count():
    answer = "First point here. Second point here. Third point that must survive."
    prompt = "Summarize the following passage in exactly two sentences:\n\nSome text."
    result = enforce_summarization_constraint(answer, prompt)
    assert result == answer


def test_enforce_summarization_constraint_never_trims_over_bullet_count():
    answer = (
        "- This first bullet point is deliberately far too long and exceeds any word limit easily\n"
        "- Second bullet is short\n"
        "- Third bullet is also reasonably short here\n"
        "- A fourth bullet that must not be dropped"
    )
    prompt = (
        "Summarize the following passage in exactly three bullet points, "
        "each no longer than 5 words:\n\nSome text."
    )
    result = enforce_summarization_constraint(answer, prompt)
    assert result == answer
    assert len(result.splitlines()) == 4


def test_enforce_summarization_constraint_only_strips_whitespace():
    answer = "  A summary with surrounding whitespace.  "
    prompt = "Summarize this in one sentence."
    assert enforce_summarization_constraint(answer, prompt) == answer.strip()
