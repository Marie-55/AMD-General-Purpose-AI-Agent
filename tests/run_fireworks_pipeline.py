"""Real Fireworks API smoke runner for local development.

This uses the same pipeline as the submitted container and makes real Fireworks
calls. It intentionally does not mock the API and does not load a bundled .env;
export FIREWORKS_API_KEY, FIREWORKS_BASE_URL, and ALLOWED_MODELS before running.
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import run_pipeline


REQUIRED_ENV = ("FIREWORKS_API_KEY", "FIREWORKS_BASE_URL", "ALLOWED_MODELS")


def require_env() -> None:
    missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            f"Missing required environment variable(s): {joined}. "
            "Export the same variables the Track 1 harness injects before running."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the real Fireworks pipeline against a local tasks JSON file.")
    parser.add_argument("--input", default="tests/sample_tasks.json", help="Path to a tasks.json-style input file")
    parser.add_argument("--output", default="", help="Optional output path. Defaults to a temporary results.json")
    args = parser.parse_args()

    require_env()

    tmp = tempfile.mkdtemp(prefix="agent-fireworks-")
    try:
        input_dir = os.path.join(tmp, "input")
        output_dir = os.path.join(tmp, "output")
        os.makedirs(input_dir)
        os.makedirs(output_dir)
        input_path = os.path.join(input_dir, "tasks.json")
        output_path = args.output or os.path.join(output_dir, "results.json")
        shutil.copyfile(args.input, input_path)

        results, metrics = run_pipeline(input_path, output_path)
        print("REAL FIREWORKS SMOKE RUN COMPLETED")
        print(json.dumps(metrics.summary(), indent=2))
        print(f"Results written to: {output_path}")
        print(json.dumps(results, indent=2))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()