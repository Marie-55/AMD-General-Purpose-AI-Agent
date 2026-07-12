"""
In-memory stand-in for utils.fireworks_client.FireworksClient.

Returns a canned answer keyed off keywords in the outgoing system/user
messages, so the golden pipeline test can exercise every category's
routing/verification/fallback logic without a network call or an API key.
This is a smoke fixture, not a quality benchmark -- it only needs to look
enough like a real completion to satisfy each category's shape checks
(JSON array for NER, a python code fence for code tasks, an "Answer:" line
for logic, etc.).
"""


def _combined_text(messages: list) -> str:
    return " ".join(m.get("content", "") for m in messages).lower()


class FakeFireworksClient:
    def __init__(self):
        self.calls = []

    def call(self, model, messages, max_tokens=400, temperature=0.2, timeout=25):
        self.calls.append({"model": model, "messages": messages, "max_tokens": max_tokens})
        combined = _combined_text(messages)
        answer = self._answer_for(combined)
        lat = {
            "model": model,
            "ttft_ms": 5.0,
            "total_latency_ms": 10.0,
            "prompt_tokens": 20,
            "completion_tokens": 20,
            "total_tokens": 40,
            "error": None,
        }
        return answer, lat

    @staticmethod
    def _answer_for(combined: str) -> str:
        if "json array" in combined and "entit" in combined:
            return '[{"text": "Barack Obama", "type": "PERSON"}, {"text": "Paris", "type": "LOCATION"}]'

        if "bug explanation" in combined or ("issues" in combined and "corrected_parts" in combined):
            return (
                "## Bug Explanation\n"
                "- The function subtracts instead of adding.\n\n"
                "## Corrected Code\n"
                "```python\n"
                "def add(a, b):\n"
                "    return a + b\n"
                "```"
            )

        if "python" in combined and (
            "code block" in combined or "code generator" in combined or "```python" in combined
        ):
            return "```python\nprint(42)\n```"

        if "sentiment" in combined:
            return "positive: the review praises the battery and camera."

        if "summar" in combined:
            return "Revenue grew steadily across all regions this quarter."

        if "clue" in combined or "puzzle" in combined:
            return "Working through the clues, Alice sits to the left of Bob.\n\nAnswer: Alice, Bob, Carol."

        return (
            "Rainbows form when sunlight is refracted and reflected inside water "
            "droplets, which splits the light into its component colors."
        )
