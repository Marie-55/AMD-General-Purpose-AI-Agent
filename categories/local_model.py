"""
local_model.py -- local GGUF inference via llama-cpp-python.

Loads a GGUF model ONCE at startup as a module-level singleton and exposes
helpers for zero-shot classification, sentiment analysis, and NER.

Thread safety
-------------
llama-cpp-python's Llama object is NOT thread-safe for concurrent calls.
All inference goes through LocalModelSingleton.generate(), which holds a
threading.Lock for the duration of each call.  Fireworks threads queue up
waiting for the lock; this is intentional and safe — local model calls are
fast (~1-3 s on CPU) so the queue drains quickly.

Environment awareness
---------------------
n_threads, n_ctx, and n_gpu_layers are read from categories.executor at
construction time so settings automatically match the runtime environment
(local dev vs AMD notebook vs Docker eval container).
"""

import logging
import os
import re
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_CATEGORIES = frozenset(
    {
        "factual_knowledge",
        "math_reasoning",
        "sentiment",
        "summarization",
        "ner",
        "code_debugging",
        "logic_puzzle",
        "code_generation",
    }
)

# Qwen2.5-1.5B is primary: stronger at structured JSON output (NER) and
# more consistent at following format instructions than SmolLM2-1.7B.
# SmolLM2-1.7B stays as fallback in case the Qwen file is missing.
_DEFAULT_PRIMARY_PATH  = "/app/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
_DEFAULT_FALLBACK_PATH = "/app/models/smollm2-1.7b-instruct-q4_k_m.gguf"

# Strips leading sentiment-question preambles so the model only sees the
# review text, not the meta-instruction.
_SENTIMENT_PREAMBLE_RE = re.compile(
    r"^\s*(?:"
    r"what\s+is\s+the\s+sentiment\s+of\s+(?:this\s+)?(?:review|text|sentence|following|passage)?|"
    r"classify\s+the\s+sentiment(?:\s+of\s+(?:this\s+)?(?:review|text|sentence|following|the\s+text))?|"
    r"label\s+the\s+sentiment(?:\s+of\s+(?:this\s+)?(?:review|text|sentence|passage))?|"
    r"analyze\s+the\s+sentiment\s+of\s+(?:this\s+)?(?:review|text|sentence|following)?|"
    r"determine\s+the\s+sentiment\s+of\s+(?:this\s+)?(?:review|text|sentence|following)?"
    r")[:\-\s]*(?:and\s+(?:briefly\s+)?justify\s+your\s+answer[?.]?\s*)?",
    re.IGNORECASE,
)

# Locates the actual text body after common NER framing phrases.
_NER_CONTEXT_RE = re.compile(
    r"(?:from the text below|in (?:the )?(?:following )?(?:text|sentence|paragraph)|"
    r"found here|in this sentence|from the following)[:\-\s]*(.+)",
    re.IGNORECASE | re.DOTALL,
)


# ---------------------------------------------------------------------------
# Singleton class
# ---------------------------------------------------------------------------

class LocalModelSingleton:
    """
    Wraps a llama_cpp.Llama instance.

    - Tries primary_path first; silently falls back to fallback_path.
    - If both fail, is_loaded() returns False and all helpers return "".
    - A threading.Lock serialises every generate() call so the underlying
      C++ object is never accessed concurrently.
    """

    def __init__(
        self,
        primary_path: str,
        fallback_path: str,
        n_ctx: int = 2048,
        n_threads: int = 2,
        n_gpu_layers: int = 0,
        verbose: bool = False,
    ) -> None:
        self._llm = None
        self._loaded_path: Optional[str] = None
        self._lock = threading.Lock()          # ← serialises all generate() calls

        for path, label in ((primary_path, "primary"), (fallback_path, "fallback")):
            if not os.path.isfile(path):
                logger.warning("Local model %s path not found: %s", label, path)
                continue
            try:
                # Deferred import: rest of codebase can be imported even when
                # llama-cpp-python is not installed (unit-test / CI environments).
                from llama_cpp import Llama  # type: ignore

                logger.info(
                    "Loading local model from %s (%s) — n_ctx=%d n_threads=%d n_gpu_layers=%d",
                    path, label, n_ctx, n_threads, n_gpu_layers,
                )
                self._llm = Llama(
                    model_path=path,
                    n_ctx=n_ctx,
                    n_threads=n_threads,
                    n_gpu_layers=n_gpu_layers,
                    verbose=verbose,
                )
                self._loaded_path = path
                logger.info("Local model loaded successfully: %s", path)
                break
            except Exception as exc:
                logger.error("Failed to load local model from %s: %s", path, exc)

        if self._llm is None:
            logger.error(
                "No local model could be loaded (primary=%s, fallback=%s). "
                "classify_with_local_model / run_sentiment / run_ner will be unavailable.",
                primary_path,
                fallback_path,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_loaded(self) -> bool:
        """Return True if a GGUF model was successfully loaded."""
        return self._llm is not None

    def generate(
        self,
        prompt_str: str,
        max_tokens: int = 200,
        temperature: float = 0.1,
        stop: Optional[list] = None,
    ) -> str:
        """
        Run synchronous CPU inference and return the generated text (stripped).

        This method acquires a threading.Lock for its entire duration so that
        concurrent callers queue up safely instead of racing on the underlying
        C++ Llama object.
        """
        if self._llm is None:
            raise RuntimeError("Local model is not loaded; cannot run inference.")

        kwargs: dict = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "echo": False,
        }
        if stop:
            kwargs["stop"] = stop

        with self._lock:                       # ← only one thread at a time
            output = self._llm(prompt_str, **kwargs)

        text: str = output["choices"][0]["text"]
        return text.strip()


# ---------------------------------------------------------------------------
# Module-level singleton management
# ---------------------------------------------------------------------------

_instance: Optional[LocalModelSingleton] = None


def get_local_model() -> LocalModelSingleton:
    """
    Return the module-level LocalModelSingleton, creating it on first call.

    Paths are read from env vars (with Docker-image defaults):
      LOCAL_MODEL_PATH          → primary GGUF
      LOCAL_MODEL_FALLBACK_PATH → fallback GGUF

    llama.cpp settings (n_threads, n_ctx, n_gpu_layers) are read from
    categories.executor.get_llm_settings() so they automatically match the
    detected runtime environment.
    """
    global _instance
    if _instance is None:
        from categories.executor import get_llm_settings  # avoid circular at module level

        primary  = os.environ.get("LOCAL_MODEL_PATH",          _DEFAULT_PRIMARY_PATH)
        fallback = os.environ.get("LOCAL_MODEL_FALLBACK_PATH", _DEFAULT_FALLBACK_PATH)
        settings = get_llm_settings()

        _instance = LocalModelSingleton(
            primary_path  = primary,
            fallback_path = fallback,
            n_ctx         = settings["n_ctx"],
            n_threads     = settings["n_threads"],
            n_gpu_layers  = settings["n_gpu_layers"],
            verbose       = False,
        )
    return _instance


# ---------------------------------------------------------------------------
# Task helpers
# ---------------------------------------------------------------------------

def _build_chatml(system: str, user: str) -> str:
    """
    Render a ChatML prompt string compatible with SmolLM2-Instruct and
    Qwen2.5-Instruct GGUF weights.
    """
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def classify_with_local_model(
    prompt: str,
    model: LocalModelSingleton,
) -> Optional[str]:
    """
    Zero-shot classify prompt into one of the 8 fixed categories.

    Returns the matched category string, or None if the model output does not
    exactly match any known category (caller should fall back to regex).
    """
    system = (
        "You are a task classifier. Classify the user's request into exactly one of these "
        "categories: factual_knowledge, math_reasoning, sentiment, summarization, ner, "
        "code_debugging, logic_puzzle, code_generation. "
        "Reply with ONLY the category name, nothing else."
    )
    user = prompt[:400]
    prompt_str = _build_chatml(system, user)

    try:
        raw = model.generate(
            prompt_str,
            max_tokens=20,
            temperature=0.0,
            stop=["\n", " ", ",", "."],
        )
    except Exception as exc:
        logger.warning("classify_with_local_model inference failed: %s", exc)
        return None

    candidate = raw.strip().lower()
    if candidate in VALID_CATEGORIES:
        return candidate
    return None


# Strips every common sentiment-task preamble so Qwen only sees the
# opinion text, not the meta-instruction.
_SENTIMENT_STRIP_RE = re.compile(
    r"^\.?\s*(?:"
    r"(?:what\s+is\s+the\s+(?:overall\s+)?)?sentiment[\s\w,]*?[:\-]?\s*|"  # "sentiment:" / "determine the sentiment"
    r"(?:determine|classify|label|analyze|identify)\s+(?:the\s+)?(?:overall\s+)?sentiment[\s\w,]*?[:\-]?\s*|"  # "classify the sentiment"
    r"(?:determine|classify|label|analyze)\s+(?:the\s+)?(?:overall\s+)?sentiment\s+and\s+[^:]+?[:\-]?\s*"  # "...and briefly justify"
    r")[:\-]?\s*",
    re.IGNORECASE,
)


def run_sentiment(prompt: str, model: LocalModelSingleton) -> str:
    """
    Classify the sentiment of the text embedded in *prompt*.

    Returns ONLY the single label word: ``positive``, ``negative``, or
    ``mixed``.  No justification, no punctuation, no extra text.
    """
    system = (
        "Classify the sentiment of the text. "
        "Reply with ONLY one word — positive, negative, or mixed. "
        "No explanation, no punctuation, nothing else."
    )

    # Strip preamble first, then fall back to the full prompt.
    cleaned = _SENTIMENT_STRIP_RE.sub("", prompt).strip()
    if not cleaned:
        cleaned = prompt.strip()

    prompt_str = _build_chatml(system, cleaned)

    try:
        raw = model.generate(
            prompt_str,
            max_tokens=10,          # label is at most 1 word (8 chars)
            temperature=0.0,        # deterministic
            stop=["\n", ":", ".", ",", " "],  # stop at first whitespace/punct
        )
        label = raw.strip().lower().rstrip(".:,")
        # Normalise any near-miss to one of the three valid labels.
        if label.startswith("pos"):
            return "positive"
        if label.startswith("neg"):
            return "negative"
        if label.startswith("mix") or label.startswith("neu"):
            return "mixed"
        # Fallback: return whatever the model said (non-empty) so the
        # caller doesn't fall through to Fireworks unnecessarily.
        return label if label else ""
    except Exception as exc:
        logger.error("run_sentiment inference failed: %s", exc)
        return ""


def _repair_json_array(raw: str) -> str:
    """
    Attempt to recover a valid JSON array from a potentially truncated or
    partially-formed model output.

    Strategy:
    1. Strip markdown fences.
    2. Find the opening '[' and try to parse everything after it.
    3. If parsing fails (truncated), close any open braces/brackets and retry.
    4. Filter out objects that don't have both 'text' and 'type' keys with
       valid NER type values so partial objects from truncation don't pollute
       the result.
    """
    import json

    VALID_TYPES = {"PERSON", "ORG", "LOCATION", "DATE"}

    # Strip markdown fences.
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip()
    cleaned = re.sub(r"```\s*$", "", cleaned).strip()

    # Find the start of the JSON array.
    start = cleaned.find("[")
    if start == -1:
        return "[]"
    candidate = cleaned[start:]

    # Try parsing as-is first.
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, list):
            valid = [
                obj for obj in parsed
                if isinstance(obj, dict)
                and isinstance(obj.get("text"), str)
                and obj.get("type", "").upper() in VALID_TYPES
            ]
            return json.dumps(valid)
    except json.JSONDecodeError:
        pass

    # Truncation repair: count unclosed braces/brackets and close them.
    open_braces   = candidate.count("{") - candidate.count("}")
    open_brackets = candidate.count("[") - candidate.count("]")
    repaired = candidate.rstrip().rstrip(",")
    if open_braces > 0:
        repaired += "}" * open_braces
    if open_brackets > 0:
        repaired += "]" * open_brackets

    try:
        parsed = json.loads(repaired)
        if isinstance(parsed, list):
            valid = [
                obj for obj in parsed
                if isinstance(obj, dict)
                and isinstance(obj.get("text"), str)
                and obj.get("type", "").upper() in VALID_TYPES
            ]
            return json.dumps(valid)
    except json.JSONDecodeError:
        pass

    return "[]"


def run_ner(prompt: str, model: LocalModelSingleton) -> str:
    """
    Extract named entities from the text referenced by prompt.

    Isolates the passage to analyse (text after framing phrases like
    "from the text below"). Falls back to full prompt if not found.

    Returns a JSON array string of {"text":…,"type":…} objects, or "[]"
    if the model output cannot be parsed as a JSON array.
    """
    system = (
        "You are a named entity recognition system. "
        "Extract all named entities from the text. "
        "Return ONLY a JSON array. Each element must be an object with exactly "
        "two keys: 'text' (the entity string) and 'type' (one of: PERSON, ORG, LOCATION, DATE). "
        "Example output: "
        '[{"text": "Paris", "type": "LOCATION"}, {"text": "2024", "type": "DATE"}]. '
        "Do not include any explanation, markdown fences, or keys other than 'text' and 'type'."
    )

    match = _NER_CONTEXT_RE.search(prompt)
    text_to_analyze = match.group(1).strip() if match else prompt.strip()

    prompt_str = _build_chatml(system, text_to_analyze)

    try:
        # 600 tokens: enough for ~10 entities with full JSON structure,
        # plus headroom so the array is never cut off mid-object.
        raw = model.generate(prompt_str, max_tokens=600, temperature=0.0)
    except Exception as exc:
        logger.error("run_ner inference failed: %s", exc)
        return "[]"

    result = _repair_json_array(raw)
    if result == "[]":
        logger.warning("run_ner: could not extract valid JSON array from: %r", raw[:200])
    return result
