# """
# setup_local_models.py
# =====================
# Developer-only script to download the quantized GGUF model weights
# needed by the Docker image.

# DO NOT push this script's output (models/) to a public repo — the weights
# are large binary files. The Dockerfile COPY instruction expects them to
# already exist under Code/models/ at build time.

# Usage
# -----
#     pip install huggingface_hub
#     python setup_local_models.py

# What it downloads
# -----------------
# Primary  : HuggingFaceTB/SmolLM2-1.7B-Instruct  →  Q4_K_M GGUF  (~1.1 GB)
# Fallback : Qwen/Qwen2.5-1.5B-Instruct            →  Q4_K_M GGUF  (~1.0 GB)

# Both files land in  Code/models/  (next to this script's parent directory).

# Approximate total download: ~2.1 GB.
# """
# import hashlib
# import os
# import sys
# from pathlib import Path


# # ---------------------------------------------------------------------------
# # Model definitions
# # ---------------------------------------------------------------------------
# # Qwen2.5-1.5B is PRIMARY: stronger structured JSON output (NER) and better
# # instruction-following at the 1-2B scale than SmolLM2-1.7B.
# # SmolLM2-1.7B is FALLBACK: used if the Qwen file is missing or fails to load.
# MODELS = [
#     {
#         "label": "Qwen2.5-1.5B-Instruct Q4_K_M (PRIMARY)",
#         "repo_id": "bartowski/Qwen2.5-1.5B-Instruct-GGUF",
#         "filename": "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
#         "local_name": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
#         # SHA-256 of the file — update after first download if you want strict
#         # verification.  Set to None to skip hash check.
#         "sha256": None,
#     },
#     {
#         "label": "SmolLM2-1.7B-Instruct Q4_K_M (fallback)",
#         "repo_id": "bartowski/SmolLM2-1.7B-Instruct-GGUF",
#         "filename": "SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
#         "local_name": "smollm2-1.7b-instruct-q4_k_m.gguf",
#         "sha256": None,
#     },
# ]

# # ---------------------------------------------------------------------------
# # Helpers
# # ---------------------------------------------------------------------------

# def _require_hf_hub():
#     try:
#         from huggingface_hub import hf_hub_download  # noqa: F401
#     except ImportError:
#         print("huggingface_hub is not installed. Run:\n  pip install huggingface_hub",
#               file=sys.stderr)
#         sys.exit(1)


# def _sha256(path: Path, chunk: int = 1 << 20) -> str:
#     h = hashlib.sha256()
#     with open(path, "rb") as f:
#         while data := f.read(chunk):
#             h.update(data)
#     return h.hexdigest()


# def _download_model(spec: dict, models_dir: Path) -> None:
#     from huggingface_hub import hf_hub_download

#     dest = models_dir / spec["local_name"]

#     if dest.exists():
#         print(f"  [skip] {dest.name} already exists ({dest.stat().st_size // (1 << 20)} MB)")
#         if spec["sha256"]:
#             actual = _sha256(dest)
#             if actual != spec["sha256"]:
#                 print(f"  [WARN] SHA-256 mismatch for {dest.name}! "
#                       f"Expected {spec['sha256']}, got {actual}. "
#                       "Delete the file and re-run to re-download.")
#             else:
#                 print(f"  [ok]   SHA-256 verified.")
#         return

#     print(f"  Downloading {spec['label']} …")
#     print(f"    repo : {spec['repo_id']}")
#     print(f"    file : {spec['filename']}")

#     tmp_path = hf_hub_download(
#         repo_id=spec["repo_id"],
#         filename=spec["filename"],
#         local_dir=str(models_dir),
#         local_dir_use_symlinks=False,
#     )

#     # hf_hub_download saves with the original filename; rename to our canonical name.
#     downloaded = models_dir / spec["filename"]
#     if downloaded.exists() and downloaded != dest:
#         downloaded.rename(dest)
#     elif Path(tmp_path).exists() and Path(tmp_path) != dest:
#         Path(tmp_path).rename(dest)

#     size_mb = dest.stat().st_size // (1 << 20)
#     print(f"  [done] {dest.name}  ({size_mb} MB)")

#     if spec["sha256"]:
#         actual = _sha256(dest)
#         if actual != spec["sha256"]:
#             print(f"  [WARN] SHA-256 mismatch! Expected {spec['sha256']}, got {actual}.")
#         else:
#             print(f"  [ok]   SHA-256 verified.")


# # ---------------------------------------------------------------------------
# # Main
# # ---------------------------------------------------------------------------

# def main() -> None:
#     _require_hf_hub()

#     # models/ lives next to this script (i.e., Code/models/).
#     script_dir = Path(__file__).resolve().parent
#     models_dir = script_dir / "models"
#     models_dir.mkdir(exist_ok=True)

#     print(f"Downloading models into: {models_dir}\n")

#     for spec in MODELS:
#         print(f"{'='*60}")
#         print(f"Model: {spec['label']}")
#         try:
#             _download_model(spec, models_dir)
#         except Exception as exc:
#             print(f"  [ERROR] Failed to download {spec['label']}: {exc}", file=sys.stderr)
#             sys.exit(1)

#     print(f"\n{'='*60}")
#     print("All models downloaded successfully.")
#     print(f"\nFiles in {models_dir}:")
#     for f in sorted(models_dir.iterdir()):
#         print(f"  {f.name}  ({f.stat().st_size // (1 << 20)} MB)")

#     print("\nNext steps:")
#     print("  1. cd to the Code/ directory")
#     print("  2. docker build -t your-image-name .")
#     print("  3. Verify image size: docker images your-image-name")
#     print("  4. Run locally:  docker run --env-file .env "
#           "-v $(pwd)/input:/input -v $(pwd)/output:/output your-image-name")


# if __name__ == "__main__":
#     main()

"""
setup_local_models.py
=====================
Developer-only script to download the quantized GGUF model weights
needed by the Docker image.

DO NOT push this script's output (models/) to a public repo — the weights
are large binary files. The Dockerfile COPY instruction expects them to
already exist under Code/models/ at build time.

Usage
-----
    pip install huggingface_hub
    python setup_local_models.py

What it downloads
-----------------
Primary  : Qwen/Qwen2.5-3B-Instruct  →  Q4_K_M GGUF  (~2.3 GB)
Fallback : HuggingFaceTB/SmolLM2-1.7B-Instruct → Q4_K_M GGUF (~1.1 GB)

Both files land in  Code/models/  (next to this script's parent directory).

Approximate total download: ~3.4 GB.
"""
import hashlib
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
# Qwen2.5-3B is PRIMARY: stronger factual reasoning, summarization, and NER
# than 1.5B, while still fitting within 4 GB RAM when quantized to Q4_K_M.
# SmolLM2-1.7B is FALLBACK: used if the Qwen file is missing or fails to load.
MODELS = [
    {
        "label": "Qwen2.5-3B-Instruct Q4_K_M (PRIMARY)",
        "repo_id": "bartowski/Qwen2.5-3B-Instruct-GGUF",
        "filename": "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        "local_name": "qwen2.5-3b-instruct-q4_k_m.gguf",
        "sha256": None,   # no strict verification
    },
    {
        "label": "SmolLM2-1.7B-Instruct Q4_K_M (fallback)",
        "repo_id": "bartowski/SmolLM2-1.7B-Instruct-GGUF",
        "filename": "SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
        "local_name": "smollm2-1.7b-instruct-q4_k_m.gguf",
        "sha256": None,
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_hf_hub():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("huggingface_hub is not installed. Run:\n  pip install huggingface_hub",
              file=sys.stderr)
        sys.exit(1)


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while data := f.read(chunk):
            h.update(data)
    return h.hexdigest()


def _download_model(spec: dict, models_dir: Path) -> None:
    from huggingface_hub import hf_hub_download

    dest = models_dir / spec["local_name"]

    if dest.exists():
        print(f"  [skip] {dest.name} already exists ({dest.stat().st_size // (1 << 20)} MB)")
        if spec["sha256"]:
            actual = _sha256(dest)
            if actual != spec["sha256"]:
                print(f"  [WARN] SHA-256 mismatch for {dest.name}! "
                      f"Expected {spec['sha256']}, got {actual}. "
                      "Delete the file and re-run to re-download.")
            else:
                print(f"  [ok]   SHA-256 verified.")
        return

    print(f"  Downloading {spec['label']} …")
    print(f"    repo : {spec['repo_id']}")
    print(f"    file : {spec['filename']}")

    tmp_path = hf_hub_download(
        repo_id=spec["repo_id"],
        filename=spec["filename"],
        local_dir=str(models_dir),
        local_dir_use_symlinks=False,
    )

    # hf_hub_download saves with the original filename; rename to our canonical name.
    downloaded = models_dir / spec["filename"]
    if downloaded.exists() and downloaded != dest:
        downloaded.rename(dest)
    elif Path(tmp_path).exists() and Path(tmp_path) != dest:
        Path(tmp_path).rename(dest)

    size_mb = dest.stat().st_size // (1 << 20)
    print(f"  [done] {dest.name}  ({size_mb} MB)")

    if spec["sha256"]:
        actual = _sha256(dest)
        if actual != spec["sha256"]:
            print(f"  [WARN] SHA-256 mismatch! Expected {spec['sha256']}, got {actual}.")
        else:
            print(f"  [ok]   SHA-256 verified.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _require_hf_hub()

    # models/ lives next to this script (i.e., Code/models/).
    script_dir = Path(__file__).resolve().parent
    models_dir = script_dir / "models"
    models_dir.mkdir(exist_ok=True)

    print(f"Downloading models into: {models_dir}\n")

    for spec in MODELS:
        print(f"{'='*60}")
        print(f"Model: {spec['label']}")
        try:
            _download_model(spec, models_dir)
        except Exception as exc:
            print(f"  [ERROR] Failed to download {spec['label']}: {exc}", file=sys.stderr)
            sys.exit(1)

    print(f"\n{'='*60}")
    print("All models downloaded successfully.")
    print(f"\nFiles in {models_dir}:")
    for f in sorted(models_dir.iterdir()):
        print(f"  {f.name}  ({f.stat().st_size // (1 << 20)} MB)")

    print("\nNext steps:")
    print("  1. cd to the Code/ directory")
    print("  2. docker build -t your-image-name .")
    print("  3. Verify image size: docker images your-image-name")
    print("  4. Run locally:  docker run --env-file .env "
          "-v $(pwd)/input:/input -v $(pwd)/output:/output your-image-name")


if __name__ == "__main__":
    main()
