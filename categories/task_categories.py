"""
The 8 fixed task categories, as named constants.

A single source of truth for category name strings so a typo or a
copy-paste mismatch becomes an import error / NameError at call time
instead of a silent misrouting bug -- e.g. main.py once called the NER
extractor for a factual_knowledge task because both were addressed by raw
string literals that were easy to swap without either linting or the
interpreter noticing.

Every module that branches on, maps, or compares a category name should
import its constant from here rather than writing the string literal.
"""

FACTUAL_KNOWLEDGE = "factual_knowledge"
MATH_REASONING = "math_reasoning"
SENTIMENT = "sentiment"
SUMMARIZATION = "summarization"
NER = "ner"
CODE_DEBUGGING = "code_debugging"
LOGIC_PUZZLE = "logic_puzzle"
CODE_GENERATION = "code_generation"

ALL_CATEGORIES = frozenset(
    {
        FACTUAL_KNOWLEDGE,
        MATH_REASONING,
        SENTIMENT,
        SUMMARIZATION,
        NER,
        CODE_DEBUGGING,
        LOGIC_PUZZLE,
        CODE_GENERATION,
    }
)

# Category groupings used by the pipeline to pick an execution strategy.
# math_reasoning: generate Python script -> run locally -> Fireworks NL fallback.
CODE_EXEC_CATEGORIES = {MATH_REASONING}
# logic_puzzle: Fireworks NL reasoning (see categories/handlers/logic.py for the
# local-first attempt tried before this).
LOGIC_NL_CATEGORIES = {LOGIC_PUZZLE}
# Routed through code-gen + AST/exec verification.
CODE_VERIFY_CATEGORIES = {CODE_GENERATION, CODE_DEBUGGING}
