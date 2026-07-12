Version: Consolidated Guide
Purpose: Use as context for an LLM while designing, implementing, debugging, or optimizing a Track 1 submission.

This document combines:

- Official Participant Guide
- Official examples
- Runtime requirements
- Docker requirements
- Fireworks integration rules
- Local model rules
- Mentor clarifications posted after the guide
- Practical implementation recommendations

---

# 1. Overview

Track 1 consists of building a Dockerized AI Agent capable of solving general Natural Language tasks across multiple capability domains.

Unlike a chatbot, the agent is evaluated automatically by a grading harness using hidden tasks.

The container starts once, processes all tasks, writes results, and exits.

No human interaction occurs during evaluation.

---

# 2. Goal

The objective is **NOT** simply obtaining the highest accuracy.

The competition uses a two-stage ranking:

1. Pass the accuracy gate.
2. Among passing submissions, minimize Fireworks token usage.

The ideal agent therefore:

- answers correctly
- avoids unnecessary Fireworks calls
- minimizes Fireworks input/output tokens
- intelligently routes tasks

---

# 3. Competition Philosophy

The challenge simulates enterprise AI systems.

Real companies often:

- host small open-source models locally
- reserve expensive API models only when necessary

Track 1 rewards exactly this behavior.

You are encouraged to:

- use local models
- use Fireworks only when beneficial
- build an intelligent routing system

---

# 4. Evaluation Categories

Your agent is evaluated on ALL eight categories.

Missing even one category lowers overall accuracy.

## 1. Factual Knowledge

Examples:

- explain concepts
- definitions
- how things work

---

## 2. Mathematical Reasoning

Examples:

- arithmetic
- percentages
- projections
- word problems

---

## 3. Sentiment Classification

Determine sentiment.

Sometimes justification is expected.

---

## 4. Text Summarization

Condense text under formatting constraints.

Examples:

- one sentence
- exact number of words
- short summary

---

## 5. Named Entity Recognition

Extract entities such as:

- Person
- Organization
- Location
- Date

---

## 6. Code Debugging

Given buggy code:

- identify issue
- provide corrected implementation

---

## 7. Logical / Deductive Reasoning

Constraint-based reasoning.

Examples:

- puzzles
- elimination
- inference

---

## 8. Code Generation

Generate working code from a specification.

---

# 5. Hidden Evaluation

The real tasks are NEVER published.

Only example schemas are provided.

The hidden evaluation consists of 19 fixed tasks.

The exact prompts are unknown.

---

# 6. Accuracy Gate (Mentor Clarification)

Originally the guide only mentioned an "accuracy threshold."

Mentors later clarified:

**The threshold is 80%.**

If your submission scores below 80%:

- you are excluded from leaderboard ranking
- token count becomes irrelevant

Passing the gate is mandatory before optimizing tokens.

---

# 7. Why Scores Look Strange

Mentor clarification:

There are exactly **19 evaluation tasks**.

Therefore accuracy percentages are discrete values:

n / 19

Examples:

84.2%

78.9%

73.7%

etc.

These are expected.

---

# 8. LLM Judge

Accuracy is determined using an LLM Judge.

The judge evaluates:

- correctness
- intent
- quality

Mentors also clarified:

The judge is **not perfectly deterministic**.

Two identical submissions may occasionally receive slightly different scores.

This is expected behavior.

---

# 9. Scoring

Stage 1

Accuracy Gate.

Must achieve at least:

80%

Stage 2

Among all passing teams:

Rank = ascending Fireworks tokens.

Lower token count = better rank.

---

# 10. Fireworks Token Counting

Only API calls routed through:

FIREWORKS_BASE_URL

count toward token usage.

Calls elsewhere are ignored.

---

# 11. Local Models

This is one of the most important rules.

Local models:

- ARE allowed
- count fully toward accuracy
- consume ZERO Fireworks tokens

Therefore:

Correct local answer = best possible token efficiency.

---

# 12. Local Model Constraints

Evaluation environment:

RAM:

4 GB

CPU:

2 vCPU

Recommended:

2B–3B models

4-bit quantized

A 7B 4-bit model nearly exhausts memory.

No Ollama is installed.

Bundle any local model directly inside the Docker image.

Compressed image must remain under:

10 GB

---

# 13. Fireworks API

Official Fireworks models are accessed remotely.

Do NOT self-host them.

Fireworks provides OpenAI-compatible APIs.

Use:

OpenAI SDK

with

base_url = FIREWORKS_BASE_URL

---

# 14. Environment Variables

Injected by the harness.

Never hardcode.

Required:

FIREWORKS_API_KEY

FIREWORKS_BASE_URL

ALLOWED_MODELS

Read ALLOWED_MODELS at runtime.

Never hardcode model names.

---

# 15. Allowed Fireworks Models

The exact IDs come from:

ALLOWED_MODELS

Typical launch-day list includes:

- minimax-m3
- kimi-k2p7-code
- gemma-4-31b-it
- gemma-4-26b-a4b-it
- gemma-4-31b-it-nvfp4

Treat these only as examples.

Always use the environment variable.

---

# 16. Gemma Clarification (Mentor)

Gemma is fully allowed.

However:

Fireworks hosts Gemma on-demand.

A 404 error means:

NOT DEPLOYED

NOT banned.

Deploy the model first if needed.

Recommended deployment:

Gemma 4 E4B

Approximate cost:

~$7/hour

Important:

Even idle deployments consume credits.

Always undeploy when finished testing.

Gemma is NOT required to pass the competition.

---

# 17. Fireworks Credits

Each team receives:

$50 Fireworks credits

One coupon code.

Sent to the registered team email.

May take:

2–3 business days.

Redeem at:

https://app.fireworks.ai/fire-pass

---

# 18. AMD Compute

There are NO separate AMD cloud credits.

Instead, every team receives:

An AMD notebook instance.

Location:

https://notebooks.amd.com/hackathon

Availability:

8 hours/day

The notebook is primarily a development environment.

---

# 19. Container Workflow

The grading harness:

1. Pulls Docker image
2. Mounts /input/tasks.json
3. Injects environment variables
4. Runs container
5. Waits
6. Reads /output/results.json
7. Counts Fireworks tokens
8. Sends outputs to LLM Judge

---

# 20. Input Format

Container reads:

/input/tasks.json

Example:

[
{
"task_id":"t1",
"prompt":"..."
}
]

---

# 21. Output Format

Write:

/output/results.json

Example:

[
{
"task_id":"t1",
"answer":"..."
}
]

---

# 22. Runtime Constraints

Startup:

under 60 seconds

Maximum runtime:

10 minutes

Maximum response per request:

30 seconds

Exit code:

0

Responses:

English only

---

# 23. Docker Constraints

Public registry.

Examples:

Docker Hub

GitHub Container Registry

Image:

linux/amd64

Compressed image:

≤10 GB

---

# 24. Submission Limits

Maximum:

10 submissions/hour/team

---

# 25. Development Strategy

No official dataset exists.

You must build your own benchmark prompts.

Suggested benchmark per category:

- factual
- math
- sentiment
- summarization
- NER
- debugging
- logic
- code generation

Benchmark every allowed model.

Determine:

- accuracy
- latency
- token usage

Then design a routing policy.

---

# 26. Optimization Strategy

Recommended order:

1. Achieve >80% accuracy.
2. Reduce Fireworks usage.
3. Shorten prompts.
4. Shorten outputs.
5. Improve routing.

Premature token optimization often reduces accuracy.

---

# 27. Practical Routing Philosophy

Possible architecture:

Input

↓

Task Classifier

↓

Decision

↓

Local Model

or

Fireworks Model

↓

Answer

A good router minimizes expensive calls while preserving accuracy.

---

# 28. Mentor Tip

While waiting for leaderboard updates:

Your Docker registry download counter

(GitHub Packages / Docker Hub)

shows whether the organizers have already pulled your image.

---

# 29. Common Failure Reasons

PULL_ERROR

Docker image inaccessible.

RUNTIME_ERROR

Program crashed.

TIMEOUT

Exceeded runtime.

OUTPUT_MISSING

results.json missing.

INVALID_RESULTS_SCHEMA

Wrong JSON format.

MODEL_VIOLATION

Called forbidden model.

IMAGE_TOO_LARGE

Docker image exceeds limit.

ACCURACY_GATE_FAILED

Below 80%.

ZERO_API_CALLS

Not a failure.

Simply means all inference was local.

---

# 30. Key Takeaways

Priority order:

1. Pass 80% accuracy.
2. Prefer local inference whenever reliable.
3. Use Fireworks only when needed.
4. Route intelligently.
5. Minimize Fireworks tokens.
6. Keep prompts concise.
7. Keep outputs concise.
8. Respect Docker/runtime constraints.
9. Read all environment variables at runtime.
10. Never hardcode evaluation assumptions.