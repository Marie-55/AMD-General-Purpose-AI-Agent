"""
Tunable runtime parameters: hard time limits, per-category token budgets,
and Fireworks role-routing tables.

Kept separate from the logic that consumes them (main.py, categories/*.py)
so a parameter change never requires touching business logic, and so every
number that affects accuracy/latency/token spend lives in one place instead
of being sprinkled across the files that happen to use it.
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
# Hard limits from the competition rules / per-request budgets
# ---------------------------------------------------------------------------
TOTAL_TIME_BUDGET_S = 9 * 60
PER_REQUEST_TIMEOUT_S = 25
LOGIC_TIMEOUT_S = 50  # logic puzzles need full CoT; 25s truncated t19
CODE_EXEC_TIMEOUT_S = 8.0

# Prompt compression threshold: only compress user messages longer than this
# many words.
COMPRESS_THRESHOLD = 300

# ---------------------------------------------------------------------------
# Budgets for calls that sit outside the per-category table below because
# they're not "the answer" itself -- they're an intermediate step (generate
# code, then fix code) that still goes through the same reasoning-capable
# model and pays the same invisible "thinking" tax before any visible
# output. All three of these were previously hardcoded small values (400 /
# 600) directly at their call sites and were observed, in real runs, being
# fully consumed by minimax-m3's internal reasoning pass before producing
# any content (fireworks_client.py's reasoning_exhausted_budget case) --
# for math codegen specifically, this caused a silent fall-through to the
# unverified NL fallback path and a wrong final answer. Centralized here so
# the next occurrence of this bug class is a config change, not an
# archaeology exercise.
MATH_CODEGEN_MAX_TOKENS = 1500
CODE_FIX_MAX_TOKENS = 1500

# ---------------------------------------------------------------------------
# Fireworks max_tokens budget per category
# ---------------------------------------------------------------------------
# Rationale per category (all >= 500 to avoid truncation unless the category
# genuinely only ever needs a short answer):
#   factual_knowledge  — 400: covers multi-part "distinguish X and Y" questions
#                        (e.g. compare two things across several dimensions)
#                        without truncating a required point; a short answer
#                        still only spends a fraction of this budget
#   math_reasoning     — step-by-step working + final answer; 700
#   sentiment          — single label + one-clause justification is short,
#                        but minimax-m3 (the only consistently healthy
#                        allowed model in practice) is a reasoning model
#                        that burns tokens on an invisible "thinking" pass
#                        before any visible content -- observed up to ~300
#                        reasoning chars (~80-100 tokens) before content
#                        even starts. 60 was tight enough to occasionally
#                        exhaust the whole budget on reasoning alone
#                        (fireworks_client.py's reasoning_exhausted_budget
#                        error) or truncate the answer mid-sentence right
#                        where it would acknowledge the second side of a
#                        mixed review; 220 gives real headroom for both.
#   summarization      — 1 sentence / 20 words / 2 sentences would need very
#                        little, but minimax-m3's reasoning pass over a full
#                        source passage can be long: observed 3933 reasoning
#                        chars (~1000 tokens) consumed with zero content left
#                        against an 800 budget, producing a total failure
#                        ("I could not determine a reliable answer."). 1800
#                        gives real headroom above the worst reasoning burn
#                        seen so far, not just the old budget's ~2x.
#   ner                — JSON array; 600 covers ~15 entities safely
#   code_debugging     — bug explanation + full corrected function; 1024
#   logic_puzzle       — reasoning trace + final answer; 4096 avoids truncation
#   code_generation    — full function implementation; 4096 (minimax spends
#                        ~1000-1500 on CoT; need headroom for the full function)
CATEGORY_MAX_TOKENS: dict[str, int] = {
    FACTUAL_KNOWLEDGE: 400,
    MATH_REASONING: 700,
    SENTIMENT: 220,
    SUMMARIZATION: 1800,
    NER: 600,
    CODE_DEBUGGING: 1024,
    LOGIC_PUZZLE: 4096,
    CODE_GENERATION: 4096,
}
DEFAULT_MAX_TOKENS = 600

# ---------------------------------------------------------------------------
# Fireworks role routing
# ---------------------------------------------------------------------------
# Each role lists candidate keyword-tiers in priority order. The first tier
# whose matching model passes the health probe wins. Tiers after the first
# are deliberate fallbacks, not "any allowed model" -- kept short and
# specific so we never fall back into an expensive/wrong-fit model.
ROLE_CANDIDATE_TIERS: dict[str, list[list[str]]] = {
    # Cheap categories (facts, sentiment): cheapest model first, fall back to
    # the general-purpose model (not the code specialist, which tends to
    # spend extra reasoning tokens on non-code prompts).
    "cheap_general": [
        ["gemma-4-26b-a4b-it", "a4b"],
        ["minimax-m3", "minimax"],
    ],
    "cheap_alt": [
        ["gemma-4-31b-it-nvfp4", "nvfp4"],
        ["minimax-m3", "minimax"],
    ],
    "quality_general": [
        ["gemma-4-31b-it", "a4b"],
        ["minimax-m3", "minimax"],
    ],
    "code_specialist": [
        ["kimi", "code"],
        ["minimax-m3", "minimax"],  # fallback when kimi is down
    ],
    # Summarization: try the cheapest quantised model first, escalate to the
    # full-precision Gemma if the answer is poor, then fall through to minimax.
    # Tier order MUST be cheapest -> best so route() and
    # route_summarization_next() both traverse in the right direction.
    "summarization": [
        ["gemma-4-26b-a4b-it"],  # tier-1: cheapest
        ["kimi", "code"],
        ["minimax-m3", "minimax"],  # tier-2: always-on safety net
    ],
    "reasoning_specialist": [
        ["minimax-m3", "minimax"],
    ],
    "ner_general": [
        ["gemma-4-26b-a4b-it", "a4b"],  # non-reasoning model, no budget waste
        ["minimax-m3", "minimax"],  # fallback if gemma is down
    ],
}

# Maps each of the 8 task categories to the routing role used to resolve its
# Fireworks model. "sentiment" and "factual_knowledge" only consult this via
# route() when their local-first attempt fails and they escalate.
CATEGORY_ROLE: dict[str, str] = {
    SENTIMENT: "cheap_general",
    FACTUAL_KNOWLEDGE: "cheap_general",
    SUMMARIZATION: "summarization",  # dedicated cascade: gemma-26b -> gemma-31b -> minimax
    NER: "ner_general",
    CODE_DEBUGGING: "code_specialist",
    CODE_GENERATION: "code_specialist",
    MATH_REASONING: "code_specialist",
    LOGIC_PUZZLE: "reasoning_specialist",
}
