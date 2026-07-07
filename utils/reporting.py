from __future__ import annotations

from collections import OrderedDict

from .models import CategoryAggregate, RunSummary, TaskExecutionStats


CATEGORY_DISPLAY_ORDER = [
    ("factual_knowledge", "factual"),
    ("math_reasoning", "math"),
    ("sentiment", "sentiment"),
    ("summarization", "summarization"),
    ("ner", "ner"),
    ("code_debugging", "code_debugging"),
    ("code_generation", "code_generation"),
    ("logic_puzzle", "logic"),
]


def build_run_summary(
    task_stats: list[TaskExecutionStats],
    elapsed_s: float,
    *,
    estimated_credit_spent_usd: float | None = None,
    estimated_remaining_usd: float | None = None,
) -> RunSummary:
    per_category: OrderedDict[str, CategoryAggregate] = OrderedDict()
    num_errors = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for stats in task_stats:
        total_prompt_tokens += int(stats.prompt_tokens)
        total_completion_tokens += int(stats.completion_tokens)
        if stats.error:
            num_errors += 1
        per_category.setdefault(stats.category, CategoryAggregate()).add(stats)

    total_tokens = total_prompt_tokens + total_completion_tokens
    return RunSummary(
        elapsed_s=elapsed_s,
        num_tasks=len(task_stats),
        num_errors=num_errors,
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        total_tokens=total_tokens,
        per_category=dict(per_category),
        task_stats=task_stats,
        estimated_credit_spent_usd=estimated_credit_spent_usd,
        estimated_remaining_usd=estimated_remaining_usd,
    )


def _fmt_float(value: float) -> str:
    return f"{value:.2f}"


def render_run_summary(summary: RunSummary) -> str:
    width = 64
    lines: list[str] = []
    lines.append("=" * width)
    lines.append("RUN SUMMARY")
    lines.append("=" * width)
    lines.append(f"{'elapsed_s':<24}: {_fmt_float(summary.elapsed_s)}")
    lines.append(f"{'num_tasks':<24}: {summary.num_tasks}")
    lines.append(f"{'num_errors':<24}: {summary.num_errors}")
    lines.append(f"{'total_prompt_tokens':<24}: {summary.total_prompt_tokens}")
    lines.append(f"{'total_completion_tokens':<24}: {summary.total_completion_tokens}")
    lines.append(f"{'total_tokens':<24}: {summary.total_tokens}")
    if summary.estimated_credit_spent_usd is not None:
        lines.append(f"{'estimated_credit_spent_usd':<24}: {_fmt_float(summary.estimated_credit_spent_usd)}")
    if summary.estimated_remaining_usd is not None:
        lines.append(f"{'estimated_remaining_usd':<24}: {_fmt_float(summary.estimated_remaining_usd)}")
    lines.append("-- by category --")

    for display_name, category_key in CATEGORY_DISPLAY_ORDER:
        aggregate = summary.per_category.get(category_key)
        if aggregate is None:
            continue
        lines.append(
            f"{display_name:<20} count={aggregate.count:<2} tokens={aggregate.total_tokens:<5} "
            f"avg_latency_ms={aggregate.avg_latency_ms:.1f}"
        )

    extra_categories = [key for key in summary.per_category.keys() if key not in dict(CATEGORY_DISPLAY_ORDER)]
    for category_key in extra_categories:
        aggregate = summary.per_category[category_key]
        lines.append(
            f"{category_key:<20} count={aggregate.count:<2} tokens={aggregate.total_tokens:<5} "
            f"avg_latency_ms={aggregate.avg_latency_ms:.1f}"
        )

    lines.append("=" * width)
    return "\n".join(lines)
