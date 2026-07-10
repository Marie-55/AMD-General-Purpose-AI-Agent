"""
Fireworks client wrapper.

Uses the OpenAI-compatible SDK, pointed at FIREWORKS_BASE_URL /
FIREWORKS_API_KEY read from the environment (never hardcoded, never our own
key baked into the image).

Streams the response so we can measure two separate latency numbers:
  - ttft_ms: time from request-sent to first token received (network + queue
    + prompt-processing latency).
  - total_latency_ms: time from request-sent to the full response completing
    (ttft + actual generation time).
This gives a real breakdown of "time to send/queue" vs "time the model spent
generating", not just one opaque round-trip number.
"""
import os
import time

from openai import OpenAI


class FireworksClient:
    def __init__(self):
        try:
            self.api_key = os.environ["FIREWORKS_API_KEY"]
            self.base_url = os.environ["FIREWORKS_BASE_URL"]
        except KeyError as exc:
            name = exc.args[0]
            raise RuntimeError(
                f"{name} is not set. The Track 1 harness injects FIREWORKS_API_KEY, "
                "FIREWORKS_BASE_URL, and ALLOWED_MODELS at evaluation time. For local "
                "development, export your own values in the shell before running; do not "
                "bundle a .env file or hardcode them in the image."
            ) from exc
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def call(self, model, messages, max_tokens=400, temperature=0.2, timeout=25):
        t_sent = time.time()
        metrics = {
            "model": model,
            "request_sent_ts": t_sent,
            "ttft_ms": None,
            "total_latency_ms": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "reasoning_chars": 0,
            "error": None,
        }
        try:
            stream = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                stream_options={"include_usage": True},
                timeout=timeout,
            )
            chunks = []
            reasoning_chunks = []
            first_token_ts = None
            usage = None
            for chunk in stream:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        if first_token_ts is None:
                            first_token_ts = time.time()
                            metrics["ttft_ms"] = round((first_token_ts - t_sent) * 1000, 1)
                        chunks.append(delta.content)
                    # Reasoning-capable models (e.g. minimax-m3) stream an
                    # internal "thinking" pass separately from `content`.
                    # If max_tokens runs out mid-reasoning, `content` can be
                    # completely empty even though the model was working --
                    # capture this so callers can tell the difference between
                    # "the model had nothing to say" and "it ran out of
                    # budget before it got to say anything."
                    reasoning_piece = getattr(delta, "reasoning_content", None)
                    if reasoning_piece:
                        reasoning_chunks.append(reasoning_piece)
                if getattr(chunk, "usage", None):
                    usage = chunk.usage

            answer = "".join(chunks).strip()
            metrics["reasoning_chars"] = len("".join(reasoning_chunks))
            t_end = time.time()
            if usage:
                metrics["prompt_tokens"] = usage.prompt_tokens
                metrics["completion_tokens"] = usage.completion_tokens
                metrics["total_tokens"] = usage.total_tokens
            metrics["request_end_ts"] = t_end
            metrics["total_latency_ms"] = round((t_end - t_sent) * 1000, 1)

            # Empty content but the model was clearly reasoning -- this is a
            # budget-exhaustion failure, not a "nothing to say" failure.
            # Surface it as an error so callers' retry logic actually treats
            # it differently instead of assuming a normal empty response.
            if not answer and metrics["reasoning_chars"] > 0:
                metrics["error"] = (
                    f"reasoning_exhausted_budget: {metrics['reasoning_chars']} "
                    f"reasoning chars produced, 0 content chars before max_tokens hit"
                )

            return answer, metrics

        except Exception as e:
            metrics["error"] = str(e)
            metrics["request_end_ts"] = time.time()
            metrics["total_latency_ms"] = round((metrics["request_end_ts"] - t_sent) * 1000, 1)
            return "", metrics