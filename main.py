from __future__ import annotations

import sys

from utils.agent import run_from_files_sync
from utils.config import load_runtime_config, validate_runtime_config


def main() -> int:
    try:
        config = load_runtime_config()
        validate_runtime_config(config)
        results = run_from_files_sync(config)
        print(f"Wrote {len(results)} results to {config.output_path}")
        return 0
    except Exception as exc:  # pragma: no cover - surfaced to the container logs
        print(f"Fatal error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
