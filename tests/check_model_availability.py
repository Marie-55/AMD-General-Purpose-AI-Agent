"""Manual dev script: probes each Fireworks model ID with a real API call.

Not a pytest test (deliberately not named test_*.py) -- it makes real,
billed Fireworks requests at import time, so pytest must never auto-collect
it. Run directly: python tests/check_model_availability.py, with
FIREWORKS_API_KEY / FIREWORKS_BASE_URL exported first.
"""
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["FIREWORKS_API_KEY"],
    base_url=os.environ["FIREWORKS_BASE_URL"],
)

models_to_check = [
    "accounts/fireworks/models/minimax-m3",
    "accounts/fireworks/models/kimi-k2p7-code",
    "accounts/fireworks/models/gemma-4-31b-it",
    "accounts/fireworks/models/gemma-4-26b-a4b-it",
    "accounts/fireworks/models/gemma-4-31b-it-nvfp4",
]

for model in models_to_check:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=5,
        )
        print(f"OK   {model} -> {resp.choices[0].message.content!r}")
    except Exception as e:
        print(f"FAIL {model} -> {e}")