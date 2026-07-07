from __future__ import annotations

import json
from pathlib import Path

from .models import Task, TaskResult


def read_tasks(path: Path) -> list[Task]:
    data = json.loads(path.read_text(encoding="utf-8"))
    tasks: list[Task] = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Each task must be a JSON object.")
        task_id = str(item["task_id"])
        prompt = str(item["prompt"])
        tasks.append(Task(task_id=task_id, prompt=prompt, raw=item))
    return tasks


def write_results(path: Path, results: list[TaskResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [{"task_id": result.task_id, "answer": result.answer} for result in results]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
