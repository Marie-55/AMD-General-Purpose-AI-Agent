"""
Regression tests for the math_reasoning "shown work" fix: judge_guide.md's
math rubric expects "minor arithmetic shown or implied" even when the
prompt doesn't explicitly ask for steps (e.g. T02: "How many units remain
at the end of Q3?" has no reasoning-trigger words, yet the Expected column
still requires the calculation trail). Previously the terse (non-"wants
steps") path returned a bare final number with no work at all.
"""
from categories.handlers.math_reasoning import (
    _format_shown_work,
    _split_work_and_final,
    _prompt_wants_steps,
)


def test_split_work_and_final_multi_line_output():
    output = "after discount: 120.0\nfinal: 129.6\n129.6"
    work, final_value = _split_work_and_final(output)
    assert final_value == "129.6"
    assert "after discount: 120.0" in work
    assert "final: 129.6" in work


def test_split_work_and_final_single_line_output():
    work, final_value = _split_work_and_final("816")
    assert work == ""
    assert final_value == "816"


def test_format_shown_work_includes_intermediate_steps():
    output = "after discount: 120.0\n129.6"
    result = _format_shown_work(output)
    assert "after discount: 120.0" in result
    assert result.endswith("Answer: 129.6")


def test_format_shown_work_bare_number_still_produces_an_answer_line():
    result = _format_shown_work("350")
    assert result == "Answer: 350"


def test_prompt_wants_steps_is_false_for_plain_quantitative_question():
    # Even prompts with no reasoning-trigger words ("show"/"explain"/"how"/
    # etc.) still get arithmetic shown -- that's now handled unconditionally
    # by _format_shown_work, not gated on this predicate. (Many natural math
    # prompts phrase the question as "how many/how much", which the
    # trigger-word regex treats as a steps request; this prompt avoids that.)
    prompt = "Calculate the total price after a 25% discount on a $160 item, then add 8% sales tax."
    assert _prompt_wants_steps(prompt) is False
