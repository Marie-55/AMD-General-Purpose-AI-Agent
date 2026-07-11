from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .engine import process_tasks


INPUT_PATH = Path("/input/tasks.json")
OUTPUT_PATH = Path("/output/results.json")


def resolve_input_path(raw: str | None) -> Path:
    if not raw:
        return INPUT_PATH
    path = Path(raw)
    if path.is_dir():
        candidate = path / "tasks.json"
        if candidate.exists():
            return candidate
        candidate = path / "tasks_0.json"
        if candidate.exists():
            return candidate
    return path


def resolve_output_path(raw: str | None) -> Path:
    if not raw:
        return OUTPUT_PATH
    path = Path(raw)
    if path.exists() and path.is_dir():
        return path / "results.json"
    if path.suffix:
        return path
    return path / "results.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AMD hackathon task runner")
    parser.add_argument(
        "--input",
        default=os.environ.get("INPUT_PATH", str(INPUT_PATH)),
        help="Input JSON file or directory containing tasks.json",
    )
    parser.add_argument(
        "--output",
        default=os.environ.get("OUTPUT_PATH", str(OUTPUT_PATH)),
        help="Output JSON file to write",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = resolve_input_path(args.input)
    output_path = resolve_output_path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        tasks = json.load(f)
    if not isinstance(tasks, list):
        raise ValueError("tasks.json must contain a list")

    results = process_tasks(tasks)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"Wrote {len(results)} results to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
