from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

from categories.router import TaskRouter
from categories.solvers import CategorySolvers
from utils.fireworks import FireworksClient
from utils.models import RuntimeConfig, Task, TaskResult, UsageTotals
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

    def solve_task(self, task: Task) -> TaskResult:
        decision = self.router.route(task.prompt)
        try:
            answer = self.solvers.solve(task, decision.category)
        except Exception as exc:
            answer = self.solvers.solve_fallback(task, error=exc)
        return TaskResult(task_id=task.task_id, answer=answer)

    def solve_tasks(self, tasks: list[Task]) -> list[TaskResult]:
        if not tasks:
            return []

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            return list(executor.map(self.solve_task, tasks))


async def run_pipeline_async(tasks: list[Task], config: RuntimeConfig, *, agent: TrackOneAgent | None = None) -> list[TaskResult]:
    active_agent = agent or TrackOneAgent(config)
    semaphore = asyncio.Semaphore(config.max_workers)

    async def process(task: Task) -> TaskResult:
        async with semaphore:
            return await asyncio.to_thread(active_agent.solve_task, task)

    results = await asyncio.gather(*(process(task) for task in tasks))
    return list(results)


def run_from_files_sync(config: RuntimeConfig) -> list[TaskResult]:
    tasks = read_tasks(config.input_path)
    agent = TrackOneAgent(config)
    results = agent.solve_tasks(tasks)
    write_results(config.output_path, results)
    return results
