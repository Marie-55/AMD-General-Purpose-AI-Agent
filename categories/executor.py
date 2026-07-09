"""
Executor selection + environment profile detection.

Three runtime environments are recognised:

  "docker"
      Inside the submitted evaluation container.
      Detected by: EVAL_ENV=docker (baked into the Dockerfile).
      Resources:   2 vCPU, 4 GB RAM, no GPU.

  "amd_notebook"
      AMD hackathon notebook / ROCm machine given to participants.
      Detected by: ROCM_HOME or HIP_VISIBLE_DEVICES present in env,
                   OR EVAL_ENV=amd_notebook.
      Resources:   varies, assume CPU-only conservative settings unless
                   ROCm GPU is explicitly available.

  "local"
      Developer laptop / workstation (this machine: 16 cores, 32 GB RAM,
      Nvidia T1200 4 GB VRAM).
      Detected by: neither of the above.
      Resources:   no restrictions — use more threads, more context, GPU
                   offload if LOCAL_USE_GPU=1.

ENV_PROFILE is computed ONCE at import time.  Every other module reads it
instead of doing their own detection so the logic is never duplicated.

Thread-count policy
-------------------
We always use ThreadPoolExecutor (never ProcessPoolExecutor) because the
Fireworks client object holds live network connections that are not safely
picklable across process boundaries.  Fireworks calls are I/O-bound so
threads are the right primitive.  The local model is protected by its own
threading.Lock, so concurrent threads queueing up on it is safe.
"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

# ---------------------------------------------------------------------------
# Profile detection (once at import time)
# ---------------------------------------------------------------------------

EnvProfile = Literal["docker", "amd_notebook", "local"]


def _detect_profile() -> EnvProfile:
    explicit = os.environ.get("EVAL_ENV", "").lower()
    if explicit == "docker":
        return "docker"
    if explicit == "amd_notebook":
        return "amd_notebook"

    # ROCm indicators → AMD notebook
    if os.environ.get("ROCM_HOME") or os.environ.get("HIP_VISIBLE_DEVICES"):
        return "amd_notebook"

    return "local"


ENV_PROFILE: EnvProfile = _detect_profile()

# ---------------------------------------------------------------------------
# Per-profile settings
# ---------------------------------------------------------------------------

# Fireworks worker threads (I/O-bound — can be high)
_WORKER_COUNT: dict[EnvProfile, int] = {
    "docker":       4,
    "amd_notebook": 4,
    "local":        8,
}

# llama.cpp inference threads (CPU-bound — match physical cores available)
_LLM_N_THREADS: dict[EnvProfile, int] = {
    "docker":       2,
    "amd_notebook": 2,
    "local":        6,   # leave some cores free for the OS / Fireworks threads
}

# llama.cpp context window (tokens) — larger = more RAM
_LLM_N_CTX: dict[EnvProfile, int] = {
    "docker":       2048,
    "amd_notebook": 2048,
    "local":        4096,
}

# GPU layers to offload to VRAM (0 = CPU only)
# On local machine: LOCAL_USE_GPU=1 enables partial offload onto T1200 (4 GB VRAM).
# ~20 layers fits comfortably; the rest stays on CPU.
def _gpu_layers() -> int:
    if ENV_PROFILE != "local":
        return 0
    if os.environ.get("LOCAL_USE_GPU", "0").lower() in ("1", "true", "yes"):
        return 20
    return 0


_LLM_N_GPU_LAYERS: int = _gpu_layers()


def get_llm_settings() -> dict:
    """Return llama.cpp constructor kwargs for the current environment."""
    return {
        "n_threads":   _LLM_N_THREADS[ENV_PROFILE],
        "n_ctx":       _LLM_N_CTX[ENV_PROFILE],
        "n_gpu_layers": _LLM_N_GPU_LAYERS,
    }


def get_max_workers() -> int:
    """Return the Fireworks thread-pool size for the current environment."""
    return _WORKER_COUNT[ENV_PROFILE]


# ---------------------------------------------------------------------------
# Executor factory
# ---------------------------------------------------------------------------

def get_task_executor(max_workers: int | None = None) -> ThreadPoolExecutor:
    """Return a ThreadPoolExecutor sized for the current environment.

    Parameters
    ----------
    max_workers:
        Override the automatic worker count.  If None, the profile default
        from ``get_max_workers()`` is used.
    """
    workers = max_workers if max_workers is not None else get_max_workers()

    print(
        f"[executor] profile={ENV_PROFILE} | workers={workers} | "
        f"llm_threads={_LLM_N_THREADS[ENV_PROFILE]} | "
        f"n_ctx={_LLM_N_CTX[ENV_PROFILE]} | "
        f"n_gpu_layers={_LLM_N_GPU_LAYERS}",
        file=sys.stderr,
    )
    return ThreadPoolExecutor(max_workers=workers)


# ---------------------------------------------------------------------------
# Legacy compat — kept so any code that imported IS_CLUSTER still works
# ---------------------------------------------------------------------------
IS_CLUSTER: bool = ENV_PROFILE in ("docker", "amd_notebook")
