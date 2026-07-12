"""Shared per-run context passed to every category handler."""
from dataclasses import dataclass
from typing import Any, Optional

from categories.metrics import MetricsCollector


@dataclass
class PipelineContext:
    client: Any
    metrics: MetricsCollector
    local_model: Optional[Any]
