# AMD General Purpose AI Agent

Track 1 implementation for the AMD Developer Hackathon.

## Layout

```text
Dockerfile
categories/
input/
main.py
README.md
requirements.txt
tests/
utils/
```

## What it does

- Reads `/input/tasks.json`
- Routes each prompt with a cheap heuristic first
- Falls back to Fireworks AI for ambiguous routing
- Uses category-specific execution paths
- Runs math and logic tasks through a local Python sandbox
- Writes `/output/results.json`

## Runtime environment

The container expects these variables at runtime:

- `FIREWORKS_API_KEY`
- `FIREWORKS_BASE_URL`
- `ALLOWED_MODELS`

For local development you can provide them in a `.env` file in the repository root.

## Run locally

```bash
python main.py
```

## Run in Docker

```bash
docker build -t amd-general-purpose-agent .
docker run --rm \
  -e FIREWORKS_API_KEY=... \
  -e FIREWORKS_BASE_URL=... \
  -e ALLOWED_MODELS=... \
  -v "$PWD/input:/input" \
  -v "$PWD/output:/output" \
  amd-general-purpose-agent
```

## Notes

- The code intentionally uses only the standard library at runtime.
- Tests are offline and do not call Fireworks.

## Tests

If you have `pytest` installed locally, you can run:

```bash
python -m pytest -q tests
```
