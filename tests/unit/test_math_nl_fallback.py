"""
Regression tests for the math NL-fallback reliability fix.

Caught in a real run: when both local and Fireworks code generation fail
completely, the pipeline fell back to a plain natural-language Fireworks
call with no code-execution result to verify the answer against -- and
produced a wrong final number (37% of 2,400 computed as 924 instead of
888) that nothing caught, because the generic math prompt has no
self-verification step. Both final-fallback branches now use a prompt that
explicitly asks the model to redo the calculation and check it.
"""
from categories.handlers.math_reasoning import _solve_math
from categories.task_categories import MATH_REASONING
from categories import prompts as P
from config import MATH_CODEGEN_MAX_TOKENS


class _FakeCodegenFailClient:
    """Never returns parseable/runnable code, forcing both the initial
    codegen attempt and the fix-retry to fail, so _solve_math reaches its
    final NL fallback branch."""

    def __init__(self):
        self.calls = []

    def call(self, model, messages, max_tokens=400, timeout=25):
        self.calls.append({"messages": messages, "max_tokens": max_tokens})
        return "I cannot help with that.", {"error": None}


ALLOWED_MODELS = "accounts/fireworks/models/minimax-m3,accounts/fireworks/models/kimi-k2p7-code"


def test_math_nl_fallback_uses_verified_prompt_when_steps_wanted(monkeypatch):
    monkeypatch.setenv("ALLOWED_MODELS", ALLOWED_MODELS)
    client = _FakeCodegenFailClient()
    prompt = "Explain how you would calculate 15% of 200."  # "explain"/"how" -> wants steps
    answer, path, model, lat = _solve_math(client, prompt, MATH_REASONING, local_model=None)

    assert path == "math_nl_fallback"
    last_system = client.calls[-1]["messages"][0]["content"]
    assert last_system == P.MATH_NL_FALLBACK_VERIFIED_SYSTEM
    assert "redo the calculation" in last_system.lower()


def test_math_nl_fallback_terse_prompt_asks_for_self_check(monkeypatch):
    monkeypatch.setenv("ALLOWED_MODELS", ALLOWED_MODELS)
    client = _FakeCodegenFailClient()
    prompt = "Calculate the total price after a 25% discount on a $160 item, then add 8% sales tax."
    answer, path, model, lat = _solve_math(client, prompt, MATH_REASONING, local_model=None)

    assert path == "math_nl_fallback"
    last_call = client.calls[-1]
    assert last_call["messages"][0]["content"] == P.MATH_TERSE_SYSTEM
    assert "redo the calculation" in P.MATH_TERSE_SYSTEM.lower()
    assert last_call["max_tokens"] == 250


def test_math_codegen_attempts_use_the_configured_token_budget(monkeypatch):
    """
    Regression guard for the actual root cause behind a wrong final answer
    in a real run: the Fireworks codegen call hardcoded max_tokens=400,
    which minimax-m3's internal reasoning pass alone could fully consume
    (observed: "1294 reasoning chars produced, 0 content chars"), leaving
    no code at all and silently falling through to the unverified NL
    fallback. Both the initial codegen attempt and the fix-retry must use
    the larger, config-driven budget.
    """
    monkeypatch.setenv("ALLOWED_MODELS", ALLOWED_MODELS)
    client = _FakeCodegenFailClient()
    prompt = "Calculate the total price after a 25% discount on a $160 item, then add 8% sales tax."
    _solve_math(client, prompt, MATH_REASONING, local_model=None)

    # calls[0] = initial codegen, calls[1] = fix-retry, calls[2] = final fallback
    assert len(client.calls) == 3
    assert client.calls[0]["max_tokens"] == MATH_CODEGEN_MAX_TOKENS
    assert client.calls[1]["max_tokens"] == MATH_CODEGEN_MAX_TOKENS
    assert MATH_CODEGEN_MAX_TOKENS > 400  # the old, too-tight budget
