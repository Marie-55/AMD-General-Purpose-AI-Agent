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
        self.api_key = os.environ.get("FIREWORKS_API_KEY", "")
        self.base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
        if not self.api_key:
            raise RuntimeError(
                "FIREWORKS_API_KEY is not set. The harness injects this at eval time; "
                "for local dev testing, export your own dev key into the same env var."
            )
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
            first_token_ts = None
            usage = None
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    if first_token_ts is None:
                        first_token_ts = time.time()
                        metrics["ttft_ms"] = round((first_token_ts - t_sent) * 1000, 1)
                    chunks.append(chunk.choices[0].delta.content)
                if getattr(chunk, "usage", None):
                    usage = chunk.usage

            answer = "".join(chunks).strip()
            t_end = time.time()
            if usage:
                metrics["prompt_tokens"] = usage.prompt_tokens
                metrics["completion_tokens"] = usage.completion_tokens
                metrics["total_tokens"] = usage.total_tokens
            metrics["request_end_ts"] = t_end
            metrics["total_latency_ms"] = round((t_end - t_sent) * 1000, 1)
            return answer, metrics

        except Exception as e:  # noqa: BLE001 - always want a metrics record, even on failure
            metrics["error"] = str(e)
            metrics["request_end_ts"] = time.time()
            metrics["total_latency_ms"] = round((metrics["request_end_ts"] - t_sent) * 1000, 1)
            return "", metrics