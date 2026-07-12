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
# Local model files.
#
#  GENERALIST : Qwen2.5-3B-Instruct Q4_K_M
#               factual / sentiment / summarization / ner / logic
#
#   CODER      : Qwen2.5-Coder-3B-Instruct Q4_K_M
#                math / debugging / code generation
#
# Only one model is ever resident at runtime (4 GB RAM constraint).
# ---------------------------------------------------------------------------
MODELS_DIR = Path(__file__).resolve().parent / "models"

def _pick(primary: str, fallback: str, env_key: str) -> str:
    """Return env override > primary (if file exists) > fallback."""
    override = os.environ.get(env_key)
    if override:
        return override
    primary_path = MODELS_DIR / primary
    if primary_path.exists() and primary_path.stat().st_size > 100_000_000:
        return str(primary_path)
    return str(MODELS_DIR / fallback)

MODEL_PATHS = {
    "generalist": _pick(
        primary="qwen2.5-3b-instruct-q4_k_m.gguf",
        fallback="qwen2.5-3b-instruct-q4_k_m.gguf",
        env_key="GENERALIST_MODEL_PATH",
    ),
    "coder": _pick(
        primary="qwen2.5-coder-3b-instruct-q4_k_m.gguf",
        fallback="qwen2.5-coder-3b-instruct-q4_k_m.gguf",
        env_key="CODER_MODEL_PATH",
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
# real constraint against the 30 s/request cap.
# ---------------------------------------------------------------------------
FACTUAL_MAX_TOKENS           = 300
SENTIMENT_MAX_TOKENS         = 120
SUMMARIZATION_MAX_TOKENS     = 300
SUMMARIZATION_RETRY_MAX_TOKENS = 450
NER_MAX_TOKENS               = 500
LOGIC_MAX_TOKENS             = 500
MATH_CODE_MAX_TOKENS         = 300
MATH_NL_MAX_TOKENS           = 250
CODE_DEBUG_MAX_TOKENS        = 500
CODE_GEN_MAX_TOKENS          = 600

FALLBACK_ANSWER = "Unable to determine a reliable answer for this task."
