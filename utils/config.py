from __future__ import annotations

import os
from pathlib import Path

from .models import RuntimeConfig


def load_env_file(path: str | Path = ".env") -> None:
    """Load a tiny .env file for local development without adding a dependency."""
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        os.environ[key] = value


def load_runtime_config() -> RuntimeConfig:
    load_env_file()

    api_key = os.getenv("FIREWORKS_API_KEY", "").strip()
    base_url = os.getenv("FIREWORKS_BASE_URL", "").strip()
    allowed_models = [model.strip() for model in os.getenv("ALLOWED_MODELS", "").split(",") if model.strip()]
    input_override = os.getenv("INPUT_PATH", "").strip()
    output_override = os.getenv("OUTPUT_PATH", "").strip()
    local_input = Path.cwd() / "input" / "tasks.json"
    local_output = Path.cwd() / "output" / "results.json"

    max_workers = max(1, int(os.getenv("MAX_WORKERS", "4")))
    request_timeout_s = max(1, int(os.getenv("REQUEST_TIMEOUT_S", "60")))
    model_timeout_s = max(1, int(os.getenv("MODEL_TIMEOUT_S", "25")))
    sandbox_timeout_s = max(1, int(os.getenv("SANDBOX_TIMEOUT_S", "4")))

    if input_override:
        input_path = Path(input_override)
    elif Path("/input/tasks.json").exists():
        input_path = Path("/input/tasks.json")
    elif local_input.exists():
        input_path = local_input
    else:
        input_path = Path("/input/tasks.json")

    if output_override:
        output_path = Path(output_override)
    elif Path("/output").exists():
        output_path = Path("/output/results.json")
    else:
        output_path = local_output

    return RuntimeConfig(
        api_key=api_key,
        base_url=base_url,
        allowed_models=allowed_models,
        input_path=input_path,
        output_path=output_path,
        max_workers=max_workers,
        request_timeout_s=request_timeout_s,
        model_timeout_s=model_timeout_s,
        sandbox_timeout_s=sandbox_timeout_s,
    )


def missing_runtime_envs(config: RuntimeConfig) -> list[str]:
    missing = []
    if not config.api_key:
        missing.append("FIREWORKS_API_KEY")
    if not config.base_url:
        missing.append("FIREWORKS_BASE_URL")
    if not config.allowed_models:
        missing.append("ALLOWED_MODELS")
    return missing


def validate_runtime_config(config: RuntimeConfig) -> None:
    missing = missing_runtime_envs(config)
    if missing:
        raise RuntimeError("Missing runtime environment variables: " + ", ".join(missing))
