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


@dataclass(frozen=True)
class RuntimeConfig:
    api_key: str
    base_url: str
    allowed_models: list[str]
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
class RouteDecision:
    category: str
    confidence: float
    source: str


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
