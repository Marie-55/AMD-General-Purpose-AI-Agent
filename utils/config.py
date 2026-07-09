from __future__ import annotations

import os
from pathlib import Path

from .models import RuntimeConfig


def load_runtime_config() -> RuntimeConfig:
    env = os.environ
    api_key = env.get("FIREWORKS_API_KEY", "").strip()
    base_url = env.get("FIREWORKS_BASE_URL", "").strip()
    allowed_models = [model.strip() for model in env.get("ALLOWED_MODELS", "").split(",") if model.strip()]
    local_model_path_raw = env.get("LOCAL_MODEL_PATH", "").strip()
    local_model_enabled = env.get("ENABLE_LOCAL_MODEL", "").strip().lower() in {"1", "true", "yes", "on"}
    local_model_path = Path(local_model_path_raw).expanduser() if local_model_path_raw else None
    local_model_n_ctx = max(256, int(env.get("LOCAL_MODEL_N_CTX", "2048")))
    local_model_n_threads = max(1, int(env.get("LOCAL_MODEL_N_THREADS", "2")))
    local_model_max_tokens = max(16, int(env.get("LOCAL_MODEL_MAX_TOKENS", "192")))
    input_override = env.get("INPUT_PATH", "").strip()
    output_override = env.get("OUTPUT_PATH", "").strip()
    local_input = Path.cwd() / "input" / "tasks.json"
    local_output = Path.cwd() / "output" / "results.json"

    max_workers = max(1, int(env.get("MAX_WORKERS", "4")))
    request_timeout_s = max(1, int(env.get("REQUEST_TIMEOUT_S", "60")))
    model_timeout_s = max(1, int(env.get("MODEL_TIMEOUT_S", "25")))
    sandbox_timeout_s = max(1, int(env.get("SANDBOX_TIMEOUT_S", "4")))

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
        local_model_path=local_model_path,
        local_model_enabled=local_model_enabled,
        local_model_n_ctx=local_model_n_ctx,
        local_model_n_threads=local_model_n_threads,
        local_model_max_tokens=local_model_max_tokens,
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
