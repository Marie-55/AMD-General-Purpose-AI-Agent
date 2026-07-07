"""Real Fireworks API smoke runner for local development.

This uses the same pipeline as the submitted container and makes real Fireworks
calls. It intentionally does not mock the API and does not load a bundled .env;
export FIREWORKS_API_KEY, FIREWORKS_BASE_URL, and ALLOWED_MODELS before running.

Local default paths are relative to the repo root:
  - input/tasks.json
  - output/results.json

The submitted Docker container still uses /input/tasks.json and
/output/results.json through main.py's defaults.
"""
import argparse
import json
import os
import sys
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
    parser = argparse.ArgumentParser(description="Run the real Fireworks pipeline against input/tasks.json.")
    parser.add_argument("--input", default="input/tasks.json", help="Path to a tasks.json-style input file")
    parser.add_argument("--output", default="output/results.json", help="Path to write results JSON")
    args = parser.parse_args()

    require_env()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    results, metrics = run_pipeline(args.input, args.output)
    print("REAL FIREWORKS SMOKE RUN COMPLETED")
    print(json.dumps(metrics.summary(), indent=2))
    print(f"Results written to: {args.output}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()