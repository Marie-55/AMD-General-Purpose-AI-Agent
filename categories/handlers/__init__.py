"""
Category dispatch table.

process_task() in main.py used to be one ~270-line if/elif chain covering
all 8 categories inline. Each category now has its own handler module with
one entry point: handle(task, category, ctx) -> {"task_id", "answer"}.
code_verify.handle serves both code_generation and code_debugging since
they share the same generate -> verify -> auto-fix -> NL-fallback pipeline.
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
from categories.handlers import (
    sentiment,
    summarization,
    factual,
    ner,
    math_reasoning,
    logic,
    code_verify,
    general,
)
from categories.handlers.context import PipelineContext

CATEGORY_HANDLERS = {
    SENTIMENT: sentiment.handle,
    SUMMARIZATION: summarization.handle,
    FACTUAL_KNOWLEDGE: factual.handle,
    NER: ner.handle,
    MATH_REASONING: math_reasoning.handle,
    LOGIC_PUZZLE: logic.handle,
    CODE_GENERATION: code_verify.handle,
    CODE_DEBUGGING: code_verify.handle,
}


def dispatch(task: dict, category: str, ctx: PipelineContext) -> dict:
    handler = CATEGORY_HANDLERS.get(category, general.handle)
    return handler(task, category, ctx)
