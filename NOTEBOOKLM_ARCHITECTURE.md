# Track 1 Agent: Cost-Aware Hybrid Routing for the AMD Developer Hackathon

## 1. The Problem

The AMD Developer Hackathon's Track 1 challenge asks teams to build a single
Docker container that acts as a general-purpose AI agent. The container is
started once, reads a fixed list of tasks from `/input/tasks.json`, answers
every task, writes `/output/results.json`, and exits. There is no human in
the loop and no retry.

Scoring happens in two stages:

1. **Accuracy gate.** A hidden set of 19 tasks is graded by an LLM judge on
   correctness, required format, and quality. A submission must score at
   least 80% (roughly 16 of 19) to be ranked at all. Below that, nothing
   else matters.
2. **Token ranking.** Among submissions that clear the gate, the ranking is
   by *ascending* Fireworks API token consumption. Local model inference is
   completely free and unlimited; only tokens spent on the Fireworks API
   count against you.

This creates a specific incentive that shapes the whole design: **a correct
answer produced locally is strictly better than the same correct answer
produced via a paid API call.** The problem isn't "how do we answer these
tasks" in isolation — it's "how do we answer these tasks as often as
possible without ever calling out to Fireworks, while never letting that
preference cause a wrong or malformed answer that would fail the judge."

## 2. The Eight Task Categories

Every task belongs to one of eight fixed categories:

1. **Factual knowledge** — explain a concept, define a term, describe how
   something works.
2. **Mathematical reasoning** — arithmetic, percentages, multi-step word
   problems.
3. **Sentiment classification** — label a piece of text's sentiment, often
   with a justification.
4. **Text summarization** — condense a passage under an exact constraint
   (a word count, a sentence count, a bullet count).
5. **Named entity recognition (NER)** — extract entities (person,
   organization, location, date, or task-specified custom types) as
   structured JSON.
6. **Code debugging** — given buggy code, find and fix the bug.
7. **Logical / deductive reasoning** — solve a puzzle from a set of clues.
8. **Code generation** — write new code from a specification.

Each category has a genuinely different risk profile for local-vs-remote
routing. A 1.5–3B local model can often produce a solid one-paragraph
factual explanation, but is much less reliable at holding together a long
chain of deductive logic, or reliably emitting syntactically valid code on
the first try. The routing strategy is built around this asymmetry rather
than treating all eight categories the same way.

## 3. Design Principle: Local-First, Verified Escalation

The agent never asks "did the local model produce *something*" — it asks
"is what the local model produced actually good enough to hand to the
judge." This distinction matters a lot in practice. A local model that
returns a technically-non-empty but wrong, truncated, or off-topic answer
is worse than useless if it's accepted uncritically, because it silently
converts a task the agent *could* have solved correctly (by escalating)
into a wrong answer the judge will penalize.

So every local-first category runs through the same shape:

1. Try the bundled local model.
2. Run its answer through a category-specific validity/confidence check —
   not a generic "is it non-empty" check, but something tailored to what
   "good" actually looks like for that category (a real `Answer:` line for
   logic puzzles, a parseable JSON array for NER, a minimum coherent length
   for summaries, and so on).
3. If it passes, accept it — zero Fireworks tokens spent.
4. If it fails, escalate to a Fireworks model chosen for that category's
   role, and every escalation path itself has a retry-on-empty-answer step
   so a single bad API response doesn't produce a blank result.

Two categories skip the "try local first" step entirely because a small
local model can't do them reliably enough to be worth attempting: code
generation and code debugging go straight to a Fireworks code-specialist
model. But even there, the agent doesn't trust the first response blindly
— generated code is parsed (to catch syntax errors) and actually executed
in a sandbox, with an automatic fix-and-retry loop, before being accepted
as final.

## 4. Math Reasoning: Verification Through Execution

Math is treated differently from the other "local-first" categories
because it has a property none of the others do: **a generated answer can
be mechanically checked.** Instead of asking a model to just state a final
number, the agent asks the local model to write a short Python script that
computes the answer, then actually runs that script in a sandboxed
subprocess and reads its printed output. If the script errors out or
produces nothing usable, the agent asks Fireworks for a script instead,
retries once with the specific error message fed back to the model if that
also fails, and only drops to an unverified natural-language numeric answer
as a last resort.

This "solve it with code you can execute, not just NL you have to trust"
pattern is a deliberate hedge against the exact failure mode you'd expect
from a small model doing multi-step arithmetic in its head: consistent,
confident, silently wrong intermediate steps.

## 5. Dynamic Fireworks Model Resolution

The set of Fireworks models actually available isn't fixed at build time —
it's injected at evaluation time via the `ALLOWED_MODELS` environment
variable, and a model listed as "allowed" isn't guaranteed to actually be
deployed and responding (a model can be technically permitted but return a
404 if it hasn't been provisioned on the account being used to grade).

To handle this without hardcoding assumptions that might be wrong on
grading day, the agent defines several abstract *roles* — a cheap
general-purpose role, a code-specialist role, a reasoning-specialist role,
a dedicated summarization cascade — each backed by an ordered list of
candidate models. At startup, every candidate across every role is
health-probed once, concurrently, with a minimal one-token request. Each
role then resolves to the first candidate in its list that actually
responded. This resolution happens exactly once per run, not once per
task, so the cost of the safety check is negligible against the total
token budget.

## 6. Guaranteed Non-Empty Answers

Every task categories path — local or remote, successful or not — funnels
through a shared "ensure non-empty answer" step before the result is
written out. A missing answer for even one task lowers overall accuracy
under the competition's grading rules, so an empty string is treated as a
strictly worse outcome than a low-confidence but present answer, and the
pipeline is built to never produce one.

## 7. Codebase Organization

The implementation deliberately separates three concerns that are easy to
tangle together and hard to untangle later:

- **Parameters** (`config.py`) — every tunable number: request timeouts,
  per-category token budgets, and the Fireworks role-routing tables. When
  a token budget needs adjusting because a model is truncating answers,
  that's a one-line change here, not a hunt through business logic.
- **Prompts** (`categories/prompts.py`) — every literal string sent to any
  model, local or remote, organized by category. Prompt wording changes
  never require touching the code that decides *when* to send them.
- **Decision logic** (`categories/handlers/`) — one module per category,
  each built on shared helpers for the try-local → check-quality →
  escalate → guarantee-non-empty pattern described above.

This split exists because the single most damaging bug found during
development came from exactly this kind of tangling: a category dispatch
that silently called the wrong handler because the category name was a
raw string literal duplicated in two places that drifted out of sync. The
current design makes category names themselves is a single source of
truth (`categories/task_categories.py`), so a typo becomes an import error
at load time instead of a silent misrouting bug discovered only by reading
production output.

## 8. Reliability Strategy, Summarized

1. Prefer local inference everywhere a small model can plausibly do the
   job, but never trust it uncritically — every local answer passes
   through an explicit, category-specific quality gate before being
   accepted.
2. Escalate to Fireworks only on a failed quality check, never
   unconditionally, and treat that escalation itself as needing a retry
   path rather than a single shot.
3. Verify anything mechanically checkable — code and math both get
   executed, not just generated — instead of trusting the model's first
   answer.
4. Guarantee output completeness (no task ever comes back with a blank
   answer) and routing correctness (category name typos become load-time
   errors, not silent misrouting) as hard invariants, not best-effort
   behavior.

## 9. Packaging and Constraints

The whole system runs inside a single Docker image (`linux/amd64`) that
bundles the local GGUF model weights directly, since the evaluation
container has no external network access to fetch them at runtime. The
target evaluation environment is deliberately modest — around 4GB RAM, 2
vCPUs, no GPU — which is why the local models are small (1.5–3B parameters,
4-bit quantized) rather than anything larger: a 7B model at that
quantization level would nearly exhaust the available memory on its own.
The harness enforces a startup time limit, a total runtime limit, and a
per-request timeout, all of which shaped the token-budget and timeout
constants throughout `config.py`.
