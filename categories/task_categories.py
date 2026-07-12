"""The 8 competition categories as named constants -- a single source of
truth so a typo'd category string becomes an import error instead of
silent misrouting.
"""
FACTUAL_KNOWLEDGE = "factual_knowledge"
MATH_REASONING = "math_reasoning"
SENTIMENT = "sentiment"
SUMMARIZATION = "summarization"
NER = "ner"
CODE_DEBUGGING = "code_debugging"
LOGIC_PUZZLE = "logic_puzzle"
CODE_GENERATION = "code_generation"

ALL_CATEGORIES = [
    FACTUAL_KNOWLEDGE,
    MATH_REASONING,
    SENTIMENT,
    SUMMARIZATION,
    NER,
    CODE_DEBUGGING,
    LOGIC_PUZZLE,
    CODE_GENERATION,
]
