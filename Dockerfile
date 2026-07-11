# ============================================================
# Stage: runtime image
# Target: linux/amd64  (evaluation harness requirement)
# ============================================================
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ------------------------------------------------------------------ #
# Python deps (everything except llama-cpp-python first so Docker    #
# cache is reused when only app code changes).                       #
# ------------------------------------------------------------------ #
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------------ #
# llama-cpp-python compilation and build tools                       #
# System build deps needed to compile llama-cpp-python from source.  #
# We install build tools, compile llama-cpp-python (forcing source   #
# build), and remove the build tools in a single step to keep lean.  #
# ------------------------------------------------------------------ #
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        gcc \
        g++ \
        libgomp1 \
    && CMAKE_ARGS="-DLLAMA_BLAS=OFF -DLLAMA_CUDA=OFF -DLLAMA_METAL=OFF -DGGML_NATIVE=OFF -DLLAMA_NATIVE=OFF" \
       pip install --no-cache-dir --no-binary llama-cpp-python "llama-cpp-python==0.3.4" \
    && apt-get purge -y build-essential cmake gcc g++ \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------ #
# Application code & models                                          #
# ------------------------------------------------------------------ #
COPY . .

# Ensure models exist to prevent silent failures if setup_local_models.py
# wasn't run before building the Docker image.
RUN test -d models && ls models/*.gguf > /dev/null 2>&1 || \
    { echo "Error: models/*.gguf not found. Run python setup_local_models.py first."; exit 1; }

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
    LOCAL_MODEL_PATH=/app/models/qwen2.5-3b-instruct-q4_k_m.gguf \
    LOCAL_MODEL_FALLBACK_PATH=/app/models/smollm2-1.7b-instruct-q4_k_m.gguf

# ------------------------------------------------------------------ #
# Entrypoint                                                         #
# ------------------------------------------------------------------ #
ENTRYPOINT ["python", "main.py"]