# ============================================================
# Fully local Track-1 agent -- no Fireworks calls, zero API tokens.
# Target: linux/amd64 (evaluation harness requirement)
# ============================================================
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake gcc g++ libgomp1 \
    && CMAKE_ARGS="-DLLAMA_BLAS=OFF -DLLAMA_CUDA=OFF -DLLAMA_METAL=OFF -DGGML_NATIVE=OFF -DLLAMA_NATIVE=OFF" \
       pip install --no-cache-dir --no-binary llama-cpp-python "llama-cpp-python==0.3.4" \
    && apt-get purge -y build-essential cmake gcc g++ \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Model weights copied as their own layers, one per file -- a single
# ~3.86GB layer (both models together, what a blanket `COPY . .` would
# produce) has no partial-resume on `docker push`: if the upload drops
# anywhere inside it, the whole layer restarts from zero. Two ~1.9GB
# layers instead means a dropped connection only costs one model's worth
# of re-upload, not both.
COPY models/qwen2.5-3b-instruct-q4_k_m.gguf models/qwen2.5-3b-instruct-q4_k_m.gguf
COPY models/qwen2.5-coder-3b-instruct-q4_k_m.gguf models/qwen2.5-coder-3b-instruct-q4_k_m.gguf

# Everything else, copied explicitly rather than `COPY . .` -- a blanket
# copy would re-touch the model files above a second time, doubling their
# footprint across layers instead of splitting it.
COPY main.py config.py setup_local_models.py ./
COPY categories/ categories/

# Fail the build early (not at runtime) if the model weights weren't
# downloaded via setup_local_models.py before `docker build`. Matches
# config.py's MODEL_PATHS exactly -- no fallback models are bundled.
RUN test -f models/qwen2.5-3b-instruct-q4_k_m.gguf && \
    test -f models/qwen2.5-coder-3b-instruct-q4_k_m.gguf || \
    { echo "Error: model files missing under models/. Run 'python setup_local_models.py' first."; exit 1; }

ENTRYPOINT ["python", "main.py"]
