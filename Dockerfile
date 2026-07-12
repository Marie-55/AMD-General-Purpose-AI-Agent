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

COPY . .

# Require at minimum one generalist and one coder model.
# The config.py _pick() function will choose the best available file at runtime.
RUN { test -f models/smollm2-1.7b-instruct-q4_k_m.gguf || \
      test -f models/phi-4-mini-instruct-q4_k_m.gguf; } || \
    { echo "Error: no generalist model found under models/."; exit 1; }
RUN { test -f models/qwen2.5-1.5b-instruct-q4_k_m.gguf || \
      test -f models/qwen2.5-coder-3b-instruct-q4_k_m.gguf; } || \
    { echo "Error: no coder model found under models/."; exit 1; }

ENTRYPOINT ["python", "main.py"]
