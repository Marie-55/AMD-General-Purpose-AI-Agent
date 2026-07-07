from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

from categories.router import TaskRouter
from categories.solvers import CategorySolvers
from utils.fireworks import FireworksClient, MockFireworksClient
from utils.models import RuntimeConfig, Task, TaskExecutionStats, TaskResult, UsageTotals
from utils.reporting import RunSummary, build_run_summary
from utils.sandbox import ExecutionSandbox
from utils.task_io import read_tasks, write_results


class TrackOneAgent:
    """Coordinate routing, category execution, and final result emission."""

    def __init__(
        self,
        config: RuntimeConfig,
        *,
        client: FireworksClient | None = None,
        router: TaskRouter | None = None,
        solvers: CategorySolvers | None = None,
        sandbox: ExecutionSandbox | None = None,
        usage: UsageTotals | None = None,
    ):
        self.config = config
        self.usage = usage or UsageTotals()
        self.client = client or FireworksClient(config)
        self.sandbox = sandbox or ExecutionSandbox(timeout_s=config.sandbox_timeout_s)
        self.router = router or TaskRouter(self.client, config, self.usage)
        self.solvers = solvers or CategorySolvers(self.client, config, self.sandbox, self.usage)

    def solve_task_with_stats(self, task: Task) -> tuple[TaskResult, TaskExecutionStats]:
        local_usage = UsageTotals()
        router = TaskRouter(self.client, self.config, local_usage)
        solvers = CategorySolvers(self.client, self.config, self.sandbox, local_usage)
        started = perf_counter()
        decision = router.route(task.prompt)
        error: str | None = None
        try:
            answer = solvers.solve(task, decision.category)
        except Exception as exc:
            error = str(exc)
            answer = solvers.solve_fallback(task, error=exc)
        elapsed_ms = (perf_counter() - started) * 1000.0
        self.usage.add(local_usage.prompt_tokens, local_usage.completion_tokens)
        result = TaskResult(task_id=task.task_id, answer=answer)
        stats = TaskExecutionStats(
            task_id=task.task_id,
            category=decision.category,
            prompt_tokens=local_usage.prompt_tokens,
            completion_tokens=local_usage.completion_tokens,
            latency_ms=elapsed_ms,
            error=error,
        )
        return result, stats

    def solve_task(self, task: Task) -> TaskResult:
        result, _ = self.solve_task_with_stats(task)
        return result

    def solve_tasks(self, tasks: list[Task]) -> tuple[list[TaskResult], list[TaskExecutionStats]]:
        if not tasks:
            return [], []

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            paired_results = list(executor.map(self.solve_task_with_stats, tasks))
        results = [result for result, _ in paired_results]
        stats = [task_stats for _, task_stats in paired_results]
        return results, stats


async def run_pipeline_async(
    tasks: list[Task],
    config: RuntimeConfig,
    *,
    agent: TrackOneAgent | None = None,
    use_mock_client: bool = False,
) -> tuple[list[TaskResult], list[TaskExecutionStats]]:
    if agent is None:
        client = MockFireworksClient(config) if use_mock_client else None
        active_agent = TrackOneAgent(config, client=client)
    else:
        active_agent = agent
    semaphore = asyncio.Semaphore(config.max_workers)

    async def process(task: Task) -> tuple[TaskResult, TaskExecutionStats]:
        async with semaphore:
            return await asyncio.to_thread(active_agent.solve_task_with_stats, task)

    results_and_stats = await asyncio.gather(*(process(task) for task in tasks))
    results = [result for result, _ in results_and_stats]
    stats = [task_stats for _, task_stats in results_and_stats]
    return results, stats


def run_from_files_sync(
    config: RuntimeConfig,
    *,
    use_mock_client: bool = False,
) -> tuple[list[TaskResult], RunSummary]:
    tasks = read_tasks(config.input_path)
    client = MockFireworksClient(config) if use_mock_client else None
    agent = TrackOneAgent(config, client=client)
    started = perf_counter()
    results, stats = agent.solve_tasks(tasks)
    elapsed_s = perf_counter() - started
    write_results(config.output_path, results)
    summary = build_run_summary(stats, elapsed_s)
    return results, summary
