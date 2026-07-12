"""All tunable constants: model files, per-category token budgets, timing
limits. Kept separate from the logic that consumes them so a parameter
change is a one-line edit here, not an archaeology exercise.
"""
import os
from pathlib import Path

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
# Competition runtime limits (context.md): startup < 60s, total < 10min,
# per-request < 30s. Budgets below leave margin under each hard cap.
# ---------------------------------------------------------------------------
TOTAL_TIME_BUDGET_S = 9 * 60
PER_TASK_SOFT_LIMIT_S = 28
CODE_EXEC_TIMEOUT_S = 8.0

# ---------------------------------------------------------------------------
# Local model files -- one per role, no fallback. A real run showed the
# smaller generic models producing broken output when used as a fallback
# (a 1.5B "coder" fallback once "fixed" a bug by introducing a TypeError,
# and once returned the original buggy code unchanged), so there is no
# quality upside to bundling a weaker backup -- it only costs image size.
#
# GENERALIST : qwen2.5-3b-instruct       -- factual / sentiment /
#              summarization / ner / logic_puzzle
# CODER      : qwen2.5-coder-3b-instruct -- math-via-code / code_debugging /
#              code_generation
#
# Only one model is ever resident at runtime (4 GB RAM constraint).
# ---------------------------------------------------------------------------
MODELS_DIR = Path(__file__).resolve().parent / "models"

MODEL_PATHS = {
    "generalist": os.environ.get(
        "GENERALIST_MODEL_PATH", str(MODELS_DIR / "qwen2.5-3b-instruct-q4_k_m.gguf")
    ),
    "coder": os.environ.get(
        "CODER_MODEL_PATH", str(MODELS_DIR / "qwen2.5-coder-3b-instruct-q4_k_m.gguf")
    ),
}

# n_ctx: 2048 is enough for all 8 task types and is faster to allocate
# than 3072. Set higher via env if a task needs more context.
N_CTX     = int(os.environ.get("LOCAL_MODEL_N_CTX",     "2048"))
N_THREADS = int(os.environ.get("LOCAL_MODEL_THREADS",   "4"))   # use all physical cores
N_BATCH   = int(os.environ.get("LOCAL_MODEL_N_BATCH",   "128")) # smaller = lower latency

# Which model each category needs.
CATEGORY_MODEL = {
    FACTUAL_KNOWLEDGE: "generalist",
    SENTIMENT:         "generalist",
    SUMMARIZATION:     "generalist",
    NER:               "generalist",
    LOGIC_PUZZLE:      "generalist",
    MATH_REASONING:    "coder",
    CODE_DEBUGGING:    "coder",
    CODE_GENERATION:   "coder",
}

# ---------------------------------------------------------------------------
# Per-category max_tokens -- kept tight because CPU generation speed is the
# real constraint against the 30 s/request cap. FACTUAL and MATH_NL were
# raised after a real run showed both truncating mid-answer: t1 (factual)
# cut off mid-sentence at 300 tokens on a multi-part "explain why" question,
# and t6 (math NL fallback) cut off at 250 tokens *before ever reaching a
# final number or an "Answer:" line* -- a guaranteed miss, not just untidy.
# ---------------------------------------------------------------------------
FACTUAL_MAX_TOKENS           = 450
SENTIMENT_MAX_TOKENS         = 120
SUMMARIZATION_MAX_TOKENS     = 300
SUMMARIZATION_RETRY_MAX_TOKENS = 450
NER_MAX_TOKENS               = 500
LOGIC_MAX_TOKENS             = 500
MATH_CODE_MAX_TOKENS         = 300
MATH_NL_MAX_TOKENS           = 450
CODE_DEBUG_MAX_TOKENS        = 500
CODE_GEN_MAX_TOKENS          = 600

# Model-assisted classification (see categories/classifier.py): only used
# when the regex pass finds a weak/no signal, so this stays tiny.
CLASSIFY_MAX_TOKENS          = 20

FALLBACK_ANSWER = "Unable to determine a reliable answer for this task."
