# AMD General Purpose AI Agent

Track 1 submission for the AMD Developer Hackathon.

## What it does

- Reads `/input/tasks.json`
- Runs the existing agent once per task
- Writes `/output/results.json`
- Uses the environment variables provided at runtime
- Fails fast with a clear error if required Fireworks settings are missing

## Required environment variables

Set these at runtime:

- `FIREWORKS_API_KEY`
- `FIREWORKS_BASE_URL`
- `ALLOWED_MODELS`

Place the Fireworks key in the `FIREWORKS_API_KEY` environment variable before
running the container. Do not put it in the repository, Dockerfile, source code,
or a committed `.env` file. In local shell testing:

```bash
export FIREWORKS_API_KEY="your-fireworks-key"
export FIREWORKS_BASE_URL="https://api.fireworks.ai/inference/v1"
export ALLOWED_MODELS="comma,separated,model,ids,provided,by,the,harness"
```

The agent reads model IDs only from `ALLOWED_MODELS`. For difficult reasoning,
math, code, debugging, and logic tasks, the model selector prefers an allowed
Gemma 4 E4B model when one appears in that list. Simpler routing, factual,
sentiment, summarization, and NER paths continue to prefer lighter models or
deterministic local solvers.

Optional runtime controls:

- `USE_MOCK_CLIENT=1`
- `ENABLE_LOCAL_MODEL=1`
- `LOCAL_MODEL_PATH=/models/local-model.gguf`
- `LOCAL_MODEL_N_CTX=2048`
- `LOCAL_MODEL_N_THREADS=2`
- `LOCAL_MODEL_MAX_TOKENS=192`
- `INPUT_PATH=/input/tasks.json`
- `OUTPUT_PATH=/output/results.json`
- `MAX_WORKERS=4`

## Input format

`/input/tasks.json` must contain a JSON array of objects:

```json
[
  {
    "task_id": "t1",
    "prompt": "..."
  }
]
```

## Output format

`/output/results.json` is written as a JSON array of objects:

```json
[
  {
    "task_id": "t1",
    "answer": "..."
  }
]
```

No extra fields are added.

## Build

```bash
docker buildx build --platform linux/amd64 -t amd-general-purpose-ai-agent .
```

## Run

```bash
docker run --rm \
  -e FIREWORKS_API_KEY="$FIREWORKS_API_KEY" \
  -e FIREWORKS_BASE_URL="$FIREWORKS_BASE_URL" \
  -e ALLOWED_MODELS="$ALLOWED_MODELS" \
  -v "$PWD/input:/input" \
  -v "$PWD/output:/output" \
  amd-general-purpose-ai-agent
```

If you want to enable a bundled local GGUF model, also pass the local model
variables and make sure the model file is included in the image or mounted into
the container at the path you set in `LOCAL_MODEL_PATH`.

## Local test

```bash
mkdir -p input output
cat > input/tasks.json <<'JSON'
[
  {
    "task_id": "t1",
    "prompt": "Explain what the CAP theorem states."
  }
]
JSON

export FIREWORKS_API_KEY="test-key"
export FIREWORKS_BASE_URL="https://api.fireworks.ai/inference/v1"
export ALLOWED_MODELS="accounts/fireworks/models/minimax-m3"
USE_MOCK_CLIENT=1 python main.py
cat output/results.json
```

## Test

```bash
python -m pytest -q tests
```
