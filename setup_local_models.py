"""Developer-only script to download the local GGUF models this agent uses.
Run this once before `docker build` -- the Dockerfile expects the files to
already exist under models/.

    pip install huggingface_hub
    python setup_local_models.py

Downloads (~5.9 GB total):
    generalist : Qwen2.5-3B-Instruct        (3B, Q4_K_M, ~1.9 GB) -- primary
                 SmolLM2-1.7B-Instruct      (1.7B, Q4_K_M, ~1.1 GB) -- fallback
                 factual / sentiment / summarization / ner / logic_puzzle
    coder      : Qwen2.5-Coder-3B-Instruct  (3B, Q4_K_M, ~1.9 GB) -- primary
                 Qwen2.5-1.5B-Instruct      (1.5B, Q4_K_M, ~1.0 GB) -- fallback
                 math_reasoning / code_debugging / code_generation

An earlier version of this script tried to download Phi-4-mini-instruct
from a repo_id that doesn't exist (bartowski/Phi-4-mini-instruct-GGUF --
the actual repo is bartowski/microsoft_Phi-4-mini-instruct-GGUF), which
silently failed and left main.py crashing on a missing file. config.py's
_pick() now always falls back to a model that's actually downloaded here,
so a partial/failed download degrades gracefully instead of crashing.

Only one model is ever loaded into RAM at runtime (see
categories/model_runtime.py) -- all four are bundled so config.py's _pick()
can choose whichever is available and best-suited for a given task.
"""
import sys
from pathlib import Path

MODELS = [
    {
        "label": "Qwen2.5-3B-Instruct Q4_K_M (generalist, primary)",
        "repo_id": "bartowski/Qwen2.5-3B-Instruct-GGUF",
        "filename": "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        "local_name": "qwen2.5-3b-instruct-q4_k_m.gguf",
    },
    {
        "label": "SmolLM2-1.7B-Instruct Q4_K_M (generalist, fallback)",
        "repo_id": "bartowski/SmolLM2-1.7B-Instruct-GGUF",
        "filename": "SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
        "local_name": "smollm2-1.7b-instruct-q4_k_m.gguf",
    },
    {
        "label": "Qwen2.5-Coder-3B-Instruct Q4_K_M (coder, primary)",
        "repo_id": "bartowski/Qwen2.5-Coder-3B-Instruct-GGUF",
        "filename": "Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf",
        "local_name": "qwen2.5-coder-3b-instruct-q4_k_m.gguf",
    },
    {
        "label": "Qwen2.5-1.5B-Instruct Q4_K_M (coder, fallback)",
        "repo_id": "bartowski/Qwen2.5-1.5B-Instruct-GGUF",
        "filename": "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
        "local_name": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
    },
]


def _require_hf_hub():
    try:
        from huggingface_hub import hf_hub_download  # noqa: F401
    except ImportError:
        print("huggingface_hub is not installed. Run:\n  pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)


def _download(spec: dict, models_dir: Path) -> None:
    from huggingface_hub import hf_hub_download

    dest = models_dir / spec["local_name"]
    if dest.exists():
        print(f"  [skip] {dest.name} already exists ({dest.stat().st_size // (1 << 20)} MB)")
        return

    print(f"  Downloading {spec['label']} ...")
    print(f"    repo: {spec['repo_id']}")
    print(f"    file: {spec['filename']}")
    try:
        downloaded_path = Path(
            hf_hub_download(
                repo_id=spec["repo_id"],
                filename=spec["filename"],
                local_dir=str(models_dir),
            )
        )
    except Exception as exc:
        print(
            f"  [ERROR] Could not download {spec['filename']} from {spec['repo_id']}: {exc}\n"
            f"  Double check the exact quant filename at "
            f"https://huggingface.co/{spec['repo_id']}/tree/main and update "
            f"MODELS in this script if the name differs.",
            file=sys.stderr,
        )
        raise

    if downloaded_path != dest:
        downloaded_path.rename(dest)
    print(f"  [done] {dest.name} ({dest.stat().st_size // (1 << 20)} MB)")


def main() -> None:
    _require_hf_hub()
    models_dir = Path(__file__).resolve().parent / "models"
    models_dir.mkdir(exist_ok=True)
    print(f"Downloading models into: {models_dir}\n")

    for spec in MODELS:
        print("=" * 60)
        print(f"Model: {spec['label']}")
        _download(spec, models_dir)

    print("\n" + "=" * 60)
    print("All models downloaded.")
    for f in sorted(models_dir.iterdir()):
        if f.is_file():
            print(f"  {f.name}  ({f.stat().st_size // (1 << 20)} MB)")

    print("\nNext: docker build -t <image>:<tag> .")


if __name__ == "__main__":
    main()
