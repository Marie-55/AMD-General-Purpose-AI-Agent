from __future__ import annotations

from dataclasses import replace
import os
import sys

from utils.agent import run_from_files_sync
from utils.config import load_runtime_config, missing_runtime_envs, validate_runtime_config
from utils.reporting import render_run_summary


def main() -> int:
    try:
        config = load_runtime_config()
        missing = missing_runtime_envs(config)
        use_mock_client = os.getenv("USE_MOCK_CLIENT", "").strip().lower() in {"1", "true", "yes", "on"}
        if missing and not use_mock_client:
            print(
                "Fatal error: missing Fireworks environment variables. "
                "The hackathon container injects them at runtime; for local debugging set USE_MOCK_CLIENT=1."
            )
            print("Missing: " + ", ".join(missing), file=sys.stderr)
            return 1

        if use_mock_client:
            print(
                "Missing Fireworks environment variables; running in local mock mode: "
                + ", ".join(missing)
            )
            if not config.allowed_models:
                config = replace(config, allowed_models=["mock-router", "mock-code", "mock-summary"])
        elif not missing:
            validate_runtime_config(config)
        results, summary = run_from_files_sync(config, use_mock_client=use_mock_client)
        print(f"Wrote {len(results)} results to {config.output_path}")
        print(render_run_summary(summary))
        return 0
    except Exception as exc:  # pragma: no cover - surfaced to the container logs
        print(f"Fatal error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
