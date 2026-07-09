from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class TaskCategory(str, Enum):
    FACTUAL = "factual"
    MATH = "math"
    SENTIMENT = "sentiment"
    SUMMARY = "summarization"
    NER = "ner"
    DEBUGGING = "code_debugging"
    LOGIC = "logic"
    CODE = "code_generation"
    UNKNOWN = "unknown"


class ExecutionTier(str, Enum):
    RULES = "rules"
    LOCAL_MODEL = "local_model"
    FIREWORKS = "fireworks"


@dataclass(frozen=True)
class RuntimeConfig:
    api_key: str
    base_url: str
    allowed_models: list[str]
    local_model_path: Path | None = None
    local_model_enabled: bool = False
    local_model_n_ctx: int = 2048
    local_model_n_threads: int = 2
    local_model_max_tokens: int = 192
    input_path: Path = Path("/input/tasks.json")
    output_path: Path = Path("/output/results.json")
    max_workers: int = 4
    request_timeout_s: int = 60
    model_timeout_s: int = 25
    sandbox_timeout_s: int = 4


@dataclass
class Task:
    task_id: str
    prompt: str
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskResult:
    task_id: str
    answer: str


@dataclass(frozen=True)
class TaskExecutionStats:
    task_id: str
    category: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    error: str | None = None

    @property
    def total_tokens(self) -> int:
        return int(self.prompt_tokens) + int(self.completion_tokens)


@dataclass(frozen=True)
class RouteDecision:
    category: str
    confidence: float
    source: str


@dataclass
class CategoryAggregate:
    count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms_total: float = 0.0
    errors: int = 0

    def add(self, stats: TaskExecutionStats) -> None:
        self.count += 1
        self.prompt_tokens += int(stats.prompt_tokens)
        self.completion_tokens += int(stats.completion_tokens)
        self.latency_ms_total += float(stats.latency_ms)
        if stats.error:
            self.errors += 1

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def avg_latency_ms(self) -> float:
        return self.latency_ms_total / self.count if self.count else 0.0


@dataclass(frozen=True)
class RunSummary:
    elapsed_s: float
    num_tasks: int
    num_errors: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    per_category: dict[str, CategoryAggregate]
    task_stats: list[TaskExecutionStats]
    estimated_credit_spent_usd: float | None = None
    estimated_remaining_usd: float | None = None


@dataclass
class UsageTotals:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False, compare=False)

    def add(self, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        with self._lock:
            self.prompt_tokens += int(prompt_tokens or 0)
            self.completion_tokens += int(completion_tokens or 0)
            self.calls += 1

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "calls": self.calls,
            }
