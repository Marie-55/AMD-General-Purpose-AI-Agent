# ============================================================
# Stage: runtime image
# Target: linux/amd64  (evaluation harness requirement)
# ============================================================
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ------------------------------------------------------------------ #
# System build deps needed to compile llama-cpp-python from source.  #
# We remove the build tools after installation to keep image lean.   #
# ------------------------------------------------------------------ #
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ------------------------------------------------------------------ #
# Python deps (everything except llama-cpp-python first so Docker    #
# cache is reused when only app code changes).                       #
# ------------------------------------------------------------------ #
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------------ #
# llama-cpp-python — CPU-only build with BLAS disabled so it works   #
# on the 2-vCPU / no-GPU evaluation environment.                     #
# We pin a specific version to ensure reproducibility.               #
# ------------------------------------------------------------------ #
RUN CMAKE_ARGS="-DLLAMA_BLAS=OFF -DLLAMA_CUDA=OFF -DLLAMA_METAL=OFF" \
    pip install --no-cache-dir "llama-cpp-python==0.3.4"

# ------------------------------------------------------------------ #
# Application code                                                   #
# ------------------------------------------------------------------ #
COPY . .

# ------------------------------------------------------------------ #
# Bundled local model weights.                                       #
# The models/ directory is populated on the developer machine by     #
# running:  python setup_local_models.py                             #
# That script is NOT pushed to the image build — the weights are.   #
# ------------------------------------------------------------------ #
# The COPY below copies models/ if it exists.  Docker will error if
# models/ is missing; run setup_local_models.py first.
COPY models/ /app/models/

# ------------------------------------------------------------------ #
# Environment profile + local model paths baked into the image.      #
#                                                                    #
# EVAL_ENV=docker  → executor.py picks docker profile automatically  #
#   (2 vCPU, 4 GB RAM, no GPU, 4 Fireworks threads)                 #
#                                                                    #
# LOCAL_MODEL_PATH / LOCAL_MODEL_FALLBACK_PATH are YOUR variables    #
# (not injected by the harness) so they are safe to bake in here.   #
#                                                                    #
# DO NOT set FIREWORKS_API_KEY, FIREWORKS_BASE_URL, or              #
# ALLOWED_MODELS here — those are harness-injected at eval time.    #
# ------------------------------------------------------------------ #
ENV EVAL_ENV=docker \
    LOCAL_MODEL_PATH=/app/models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
    LOCAL_MODEL_FALLBACK_PATH=/app/models/smollm2-1.7b-instruct-q4_k_m.gguf

# ------------------------------------------------------------------ #
# Entrypoint                                                         #
# ------------------------------------------------------------------ #
ENTRYPOINT ["python", "main.py"]
