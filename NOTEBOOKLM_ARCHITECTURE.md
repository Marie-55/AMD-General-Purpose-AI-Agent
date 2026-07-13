# Track 1 Agent (Fully-Local Variant): Zero-Token Routing for the AMD Developer Hackathon

## 1. The Problem, Revisited

This is a second implementation of the same Track 1 challenge covered by
the Fireworks-hybrid version of this agent (see the companion doc if you
have it). The competition's rules are unchanged: a single Docker container
reads `/input/tasks.json`, answers every task across 8 fixed categories,
writes `/output/results.json`, and is scored in two stages — an 80%
LLM-judged accuracy gate first, then ranked by ascending Fireworks token
usage among everyone who clears it.

The hybrid version treated this as a routing problem: try local, escalate
to a paid API when the local answer isn't good enough. This version starts
from a different, sharper observation instead: **leaderboard entries were
already passing the accuracy gate with exactly zero Fireworks tokens
spent.** If local-only inference can be made reliable enough, there is no
routing decision to make at all — the entire Fireworks dependency, and
every failure mode that comes with depending on a remote API (rate limits,
model deployment status, network calls inside a runtime budget), can be
removed from the design entirely.

## 2. The Core Bet

Two bundled quantized models answer all 8 categories, unconditionally, no
escalation path, no fallback API:

- **Generalist** — Qwen2.5-3B-Instruct, handling factual knowledge,
  sentiment, summarization, NER, and logic puzzles.
- **Coder** — Qwen2.5-Coder-3B-Instruct, handling math reasoning, code
  debugging, and code generation.

This is a real bet, not a default: it means every category's accuracy
ceiling is now whatever a 3B, 4-bit-quantized, CPU-only model can do,
period. There's no safety net to catch a case the local model genuinely
can't handle. The mitigation isn't "add an escape hatch" — it's "make the
local answer itself more reliable," through the same instinct that drove
the math self-consistency check and the NER grammar constraint described
below.

## 3. Why Only Two Models, Never Loaded Together

The target evaluation environment is deliberately modest: roughly 4GB RAM,
2 vCPUs, no GPU. Two 3B models at 4-bit quantization together would not
comfortably fit in memory at once alongside the Python runtime and
`llama.cpp`'s own overhead. So the two models are never resident
simultaneously — every task is classified and assigned to a model role
*before* any model actually runs, tasks are grouped by which model they
need, and each model is loaded into memory at most once per run: load the
generalist, answer everything that needs it, unload, load the coder,
answer everything that needs that. In the worst case this is exactly two
model loads for an entire batch of tasks, regardless of how many tasks
there are or what order they arrive in.

## 4. Classification Has to Run Before a Model Is Even Loaded

Deciding which of the two models a task needs requires classifying it
first — which creates a chicken-and-egg problem, because one of the two
classification strategies used here needs a model loaded to work. The
resolution: classification is hybrid, regex first, model-assisted only
when needed, and the generalist model is loaded specifically to satisfy
this need before task bucketing happens (which costs nothing extra, since
the generalist bucket needs that same model loaded anyway).

A fast regex pass scores every category by counting matched keyword/phrase
patterns. If the strongest match is specific enough (multiple independent
patterns agreeing), it's trusted outright with no further cost. But when
the signal is weak — a prompt phrased in a way the regex patterns didn't
anticipate — the already-loaded generalist model is asked directly to name
the category.

This distinction matters because pure regex classification failed
silently in exactly the way you'd expect: a prompt asking to "Return
entities grouped by type as JSON" has no "extract" or "named entity"
keyword in it, so a regex-only classifier defaulted it to
factual_knowledge — and the task came back as a malformed JSON object
instead of a proper entity array, because it never reached the NER
handler's grammar-constrained decoding at all. A logic puzzle phrased as
"Give one valid finishing order" similarly missed every logic-puzzle
pattern and came back as an unverified one-line answer that, on
inspection, actually violated one of the puzzle's own stated clues. Both
failures were invisible from the outside — the task got *an* answer, just
the wrong kind of answer, produced by the wrong code path. Hybrid
classification exists specifically to close that gap.

## 5. Math: Trust Nothing Until Two Independent Methods Agree

Math problems get solved twice, independently, by the same model:

1. Generate a short Python script that computes the answer, then actually
   execute it in a sandboxed subprocess and read what it printed.
2. Separately, generate a natural-language, step-by-step derivation of the
   same problem — no shared context with the code attempt.

If both produce a number and they agree, the code-execution answer is
returned, since it's backed by something mechanically checkable. If they
disagree, one more code attempt is generated as a tie-breaker between the
two original answers, rather than picking one arbitrarily.

The reason this exists at all: code that runs successfully is not the same
thing as code that's *correct*. A generated script can compute a
percentage wrong, execute without any error, and print a confident, clean,
completely wrong final number — execution only proves the code ran, never
that its logic actually matches the word problem. This was observed
directly during development on a multi-step inventory problem, where
generated code silently computed 37% of 2400 as 924 instead of 888 and
propagated that error cleanly through to a wrong final answer. Cross-
checking against an independently-derived natural-language answer is a way
to catch that class of error without needing a second, larger model.

A secondary guard exists in the same module: a natural-language derivation
that gets cut off before ever reaching a final number (hitting its output
token limit mid-sentence) is never treated as valid, either as the
cross-check value or as a returned answer on its own — a truncated
derivation with no conclusion is worse than useless if a number gets
scraped out of it by accident.

## 6. NER: Making Malformed Output Structurally Impossible

Named entity recognition asks a model to return strictly-formatted JSON —
exactly the kind of output smaller models are prone to getting subtly
wrong (a trailing comma, an extra explanatory sentence before the JSON, an
object instead of an array). Rather than generating freely and trying to
repair or reject bad JSON after the fact, this category constrains the
model's output at the decoder level with a formal grammar: the sampler is
restricted, token by token, to only ever produce sequences that are valid
instances of "a JSON array of `{text, type}` objects." A malformed
response isn't rejected after generation — it's never generated in the
first place, because it was never a reachable path through the grammar. A
lightweight regex-based repair step still exists as a last line of
defense, but it's a backstop, not the primary strategy.

## 7. What This Version Doesn't Have

Compared to a hybrid design, this version has no health-probing of remote
models, no per-category "role" resolution, no multi-tier escalation
cascades, no retry-with-a-different-model logic, and no dependency on
harness-injected credentials being valid at grading time. The entire
Fireworks client layer simply isn't present. What replaces all of that
complexity is a metrics layer that logs per-task elapsed time and an
estimated local token count (useful for tuning, not for competition
scoring, since local inference always counts as zero Fireworks tokens),
plus a hard total-runtime budget check: if a run is running long, whatever
tasks remain get a clearly-labeled fallback answer instead of risking a
timeout that would fail the whole submission.

## 8. Codebase Organization

The same separation-of-concerns discipline carries over from the hybrid
design: `config.py` holds every tunable number (token budgets, timeouts,
which model each category maps to), `categories/prompts.py` holds every
literal prompt string, and `categories/handlers.py` holds one function per
category built from those two. The overall codebase is meaningfully
smaller than the hybrid version specifically because an entire class of
routing/escalation/health-check logic doesn't need to exist when there's
only ever one place a given task can go.

## 9. Packaging

Both model files are downloaded once, up front, via a standalone setup
script, and baked directly into the Docker image — the evaluation
container has no network access to fetch them at runtime. `llama-cpp-python`
is deliberately compiled in its own build step, after the system
build tools it needs are installed, rather than being listed as an
ordinary Python dependency — listing it in the main requirements file
causes it to attempt a source build before a C/C++ compiler is even
available in the image, which fails outright. No fallback/backup model
files are bundled into the final image: only the two models the pipeline
actually uses are shipped, keeping the image meaningfully smaller than
bundling every candidate model that was ever considered during
development.
