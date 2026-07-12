# Architecture

Track 1 AMD Hackathon agent. Fully local: every task is answered by one of
two on-disk GGUF models, no Fireworks API calls, zero API tokens spent.

## Runtime flow

```
main.py
  1. load_tasks()                      -- /input/tasks.json (falls back to ./input/tasks.json)
  2. load "generalist" model           -- needed for classification's model-assisted fallback
  3. classify every task               -- categories/classifier.py
  4. bucket tasks by required model    -- config.CATEGORY_MODEL
  5. run the generalist bucket         -- model already loaded, no reload
  6. load "coder", run the coder bucket
  7. write_results()                   -- /output/results.json
  8. write_metrics()                   -- /output/metrics.json (see "Metrics" below)
```

Only two model loads ever happen in a run (one per bucket, skipped entirely
if a bucket is empty) -- the eval environment has 4GB RAM, not enough to
hold both models resident at once, so `ModelRuntime` only ever keeps one
loaded and tasks are grouped so each model loads exactly once.

## Models

No fallback models -- one file per role, both required at build time.

| Role | Model | File | Categories handled |
|---|---|---|---|
| `generalist` | Qwen2.5-3B-Instruct | `qwen2.5-3b-instruct-q4_k_m.gguf` | factual_knowledge, sentiment, summarization, ner, logic_puzzle |
| `coder` | Qwen2.5-Coder-3B-Instruct | `qwen2.5-coder-3b-instruct-q4_k_m.gguf` | math_reasoning, code_debugging, code_generation |

`coder` must be the code-specialized model, not a smaller generic model: a
real run showed a generic 1.5B fallback producing broken fixes in
code_debugging (once "fixing" a bug by comparing against `None`, which
raises `TypeError`; once returning the original buggy code unchanged).
There's enough time budget headroom to afford the larger, more accurate
model, so quality wins over the small latency cost.

Both files are downloaded up front by `setup_local_models.py` (run once,
before `docker build`) and baked into the image. `config.MODEL_PATHS`
resolves them by fixed filename under `models/`, overridable via the
`GENERALIST_MODEL_PATH` / `CODER_MODEL_PATH` env vars.

`categories/model_runtime.py`'s `ModelRuntime` wraps a single `llama_cpp.Llama`
instance: `load(key)` is a no-op if that model is already current, otherwise
it unloads whatever's resident and loads the requested one. `generate()`
calls `create_chat_completion()` (relies on the GGUF's built-in chat
template) and returns `""` on any generation error rather than raising, so
a single bad call degrades to a fallback answer instead of crashing the run.

## Classification

`categories/classifier.py` is a hybrid regex + model classifier, not pure
regex:

1. `classify_regex(prompt)` scores every category by counting keyword/phrase
   pattern matches and returns `(best_category, score)`.
2. If `score >= 2`, that match is specific enough to trust outright -- the
   generalist model isn't consulted at all for these (fast, and avoids an
   unnecessary generation call).
3. If `score <= 1` (weak or no signal), the already-loaded generalist model
   is asked directly to name one of the 8 categories (`prompts.CLASSIFY_SYSTEM`,
   `config.CLASSIFY_MAX_TOKENS = 20`), matched back with a word-boundary
   regex (not a plain substring check -- `"ner"` is literally a substring of
   `"generation"`, which previously misrouted `code_generation` answers to
   the NER handler).

This exists because pure regex silently defaulted to `factual_knowledge`
for phrasings its patterns didn't anticipate (e.g. "Return entities grouped
by type as JSON" has no "extract"/"named entity" keyword; "Give one valid
finishing order" has no "determine the order" phrasing) -- both got routed
through the wrong handler entirely, one producing a malformed NER output
with no grammar constraint applied, the other an unverified answer that
turned out to violate one of its own puzzle's clues. Since classification
needs the generalist model, `main.py` loads it first, before bucketing.

## Per-category handling (`categories/handlers.py`)

| Category | Model | Notes |
|---|---|---|
| `factual_knowledge` | generalist | Single call, `FACTUAL_MAX_TOKENS=450`. |
| `sentiment` | generalist | Single call; if the label isn't in the first 80 chars, a regex pass surfaces it from later in the answer rather than discarding the response. |
| `summarization` | generalist | Never mechanically trims to hit a word/sentence/bullet constraint -- compliance is prompt-driven only. Retries once (higher budget, stricter prompt) only if the first answer looks empty or cut off mid-sentence. |
| `ner` | generalist | Output is constrained by a GBNF grammar (`categories/grammars.py`) that forces a JSON array of `{"text", "type"}` objects at the sampler level. `_repair_ner_json()` is a last-resort regex-based salvage if the grammar somehow still fails to produce parseable JSON. |
| `logic_puzzle` | generalist | If the answer has no `"Answer:"` line, one terse retry asks specifically for one. |
| `math_reasoning` | coder | See "Math self-consistency" below. |
| `code_debugging` | coder | Single call. |
| `code_generation` | coder | Extracted code is `ast.parse()`-checked; a syntax error triggers one fix-retry with the error message fed back to the model. |

## Math self-consistency (`categories/math_solver.py`)

Motivation: code that executes successfully can still encode the wrong
arithmetic -- execution proves the code *ran*, not that its logic matches
the word problem (observed directly: a warehouse-inventory problem where
generated code computed 37% of 2400 as 924 instead of 888, executed
without error, and returned a confidently wrong final answer).

1. Generate a Python solution, execute it in a subprocess
   (`categories/code_exec.py::run_code_safely`, hard timeout, real kill
   since subprocess timeouts are reliably enforceable unlike interrupting
   an in-process `llama.cpp` generation).
2. Independently generate a natural-language step-by-step derivation from
   the same model.
3. If both produce a number and they agree, return the code answer
   (execution-verified). If they disagree, do one tie-breaking resample of
   the code path; if that new value matches the NL value, trust the NL
   answer instead, otherwise default to the (execution-verified) code
   answer.
4. A truncated NL derivation (cut off before a final number or an
   `"Answer:"` line) is never trusted as a cross-check or returned directly
   -- a real run showed a cutoff at "...Average speed = 20 km" with no
   final value, which would otherwise have been silently accepted with
   "20" extracted as the answer.

## Metrics

`main.py` tracks per-task elapsed time, an estimated token count (~4
chars/token, local only -- `fireworks_tokens` is always `0`), and whether
the task fell back to `config.FALLBACK_ANSWER`. A run summary (answer rate,
budget used, slow tasks) is printed to stdout and written to
`/output/metrics.json` alongside `/output/results.json`. `answer_rate`
(non-fallback / total) is a rough proxy for the accuracy gate, not the
actual LLM-judged score.

## Docker

`Dockerfile`: `python:3.11-slim`, `llama-cpp-python==0.3.4` compiled from
source (CPU-only: `-DLLAMA_BLAS=OFF -DLLAMA_CUDA=OFF -DLLAMA_METAL=OFF
-DGGML_NATIVE=OFF -DLLAMA_NATIVE=OFF`) with build tools purged afterward.
`requirements.txt` intentionally does **not** list `llama-cpp-python` --
it's compiled in a later `RUN` step, after `build-essential`/`cmake` are
installed; listing it in `requirements.txt` makes `pip install -r
requirements.txt` try to build it before those tools exist and fails.

A build-time check fails fast if either model file is missing under
`models/` (run `setup_local_models.py` first). `.dockerignore` excludes
`tests/`, `input/`, `output/`, `.git`, `.env`, and both now-unused fallback
GGUF files that may still be sitting on disk locally.

## Competition constraints this design targets

- 4GB RAM / 2 vCPU eval environment -> two 3B Q4 models, never both
  resident, `N_THREADS=4`/`N_CTX=2048` tuned for that profile.
- 30s per-request cap -> per-category `max_tokens` budgets sized to the
  observed CPU throughput, not to "avoid reasoning tax" the way a hosted
  reasoning model would need.
- 10 minute total runtime cap -> `TOTAL_TIME_BUDGET_S = 9*60`, checked
  before each task; remaining tasks get `FALLBACK_ANSWER` if exceeded
  rather than running over.
- Local inference costs zero Fireworks tokens and counts fully toward
  accuracy -- the entire point of this architecture is to answer every
  task locally at high enough quality to pass the 80% gate without ever
  needing the Fireworks fallback this project's earlier (`arch3`-era)
  hybrid architecture relied on.
