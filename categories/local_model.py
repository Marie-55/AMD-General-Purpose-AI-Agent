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
import json
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

_POSITIVE_TERMS = {
    "excellent", "fast", "great", "good", "helpful", "love", "loved",
    "smooth", "improved", "better", "best", "exceeded", "works", "worked",
    "fixed", "resolved", "successful", "satisfied", "pleasant", "unbeatable",
}
_NEGATIVE_TERMS = {
    "bad", "broken", "disaster", "frustrating", "unbearable", "slow",
    "awful", "terrible", "worst", "noise", "noisy", "issue", "problem",
    "waiting", "delay", "delayed", "failure", "failed", "bug", "hate",
}
_SENTIMENT_CONTRAST_TERMS = ("but", "however", "though", "although", "yet", "despite")

_NER_TITLE_ONLY = {
    "prime minister", "president", "minister", "governor", "mayor",
    "doctor", "dr", "dr.", "mr", "mr.", "mrs", "mrs.", "ms", "ms.",
}
_NER_ORG_HINTS = (
    "university", "institute", "college", "school", "laboratory", "lab",
    "corp", "inc", "company", "group", "bank", "agency", "ministry",
    "hospital", "nvidia", "microsoft", "unicef", "eth", "services",
)
_NER_DATE_PATTERNS = (
    re.compile(r"\b\d{1,2}\s+[A-Z][a-z]+\s+\d{4}\b"),
    re.compile(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b"),
)
_NER_YEAR_PATTERN = re.compile(r"\b\d{4}\b")
_NER_CAPITALIZED_PHRASE = re.compile(r"\b(?:[A-Z][a-z]+|[A-Z]{2,})(?:\s+(?:[A-Z][a-z]+|[A-Z]{2,}|of|and|the|for|&))*\b")
_NER_STOPWORDS = {
    "extract", "identify", "return", "json", "person", "persons", "organization",
    "organizations", "location", "locations", "date", "dates", "named", "entities",
    "entity", "every", "all", "the", "and", "or", "on", "in", "at", "from",
    "with", "by", "of", "to", "for", "as", "dr", "mr", "mrs", "ms",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
}


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

    Returns a single sentence that starts with the label (positive,
    negative, or neutral) and includes one short justification.
    """
    system = (
        "Classify the sentiment of the text. "
        "Reply with exactly one sentence starting with positive, negative, or neutral. "
        "Include a short justification in the same sentence. "
        "No bullets, no extra sentences, and no markdown."
    )

    # Strip preamble first, then fall back to the full prompt.
    cleaned = _SENTIMENT_STRIP_RE.sub("", prompt).strip()
    if not cleaned:
        cleaned = prompt.strip()

    prompt_str = _build_chatml(system, cleaned)

    def _heuristic_label(text: str) -> str:
        lowered = re.sub(r"[^a-z0-9\s]", " ", text.lower())
        tokens = lowered.split()
        if not tokens:
            return ""

        pos = sum(lowered.count(term) for term in _POSITIVE_TERMS)
        neg = sum(lowered.count(term) for term in _NEGATIVE_TERMS)

        contrast_idx = -1
        for term in _SENTIMENT_CONTRAST_TERMS:
            if term in tokens:
                contrast_idx = max(contrast_idx, tokens.index(term))

        if contrast_idx > 0:
            pre = " ".join(tokens[:contrast_idx])
            post = " ".join(tokens[contrast_idx + 1 :])
            pos = sum(pre.count(term) for term in _POSITIVE_TERMS) + 2 * sum(post.count(term) for term in _POSITIVE_TERMS)
            neg = sum(pre.count(term) for term in _NEGATIVE_TERMS) + 2 * sum(post.count(term) for term in _NEGATIVE_TERMS)

        if pos == 0 and neg == 0:
            return "neutral"
        if pos > 0 and neg > 0 and abs(pos - neg) <= 1:
            return "neutral"
        return "negative" if neg > pos else "positive"

    def _sentiment_sentence(label: str, text: str) -> str:
        lowered = re.sub(r"[^a-z0-9\s]", " ", text.lower())
        pos_terms = [term for term in _POSITIVE_TERMS if term in lowered]
        neg_terms = [term for term in _NEGATIVE_TERMS if term in lowered]

        if label == "positive":
            detail = (
                f"it emphasizes {' and '.join(pos_terms[:2])}"
                if pos_terms else "the overall tone is favorable"
            )
        elif label == "negative":
            detail = (
                f"it emphasizes {' and '.join(neg_terms[:2])}"
                if neg_terms else "the overall tone is unfavorable"
            )
        else:
            detail = "it balances praise and criticism or stays mostly even"
        return f"{label}: {detail}."

    heuristic = _heuristic_label(cleaned)
    if heuristic:
        return _sentiment_sentence(heuristic, cleaned)

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
            return _sentiment_sentence("positive", cleaned)
        if label.startswith("neg"):
            return _sentiment_sentence("negative", cleaned)
        if label.startswith("mix") or label.startswith("neu"):
            return _sentiment_sentence("neutral", cleaned)
        # Fallback: return whatever the model said (non-empty) so the
        # caller doesn't fall through to Fireworks unnecessarily.
        return label if label else _sentiment_sentence("neutral", cleaned)
    except Exception as exc:
        logger.error("run_sentiment inference failed: %s", exc)
        return _sentiment_sentence(heuristic or "neutral", cleaned)


def run_factual(prompt: str, model: LocalModelSingleton) -> str:
    """Answer factual knowledge prompts directly with the local model."""
    system = (
        "Answer the user's question directly and concisely in 2-4 sentences. "
        "Be as concise as possible and minimize output tokens while answering. "
        "No preamble, no restating the question, and no markdown unless it is clearly helpful."
    )
    prompt_str = _build_chatml(system, prompt.strip())

    try:
        raw = model.generate(
            prompt_str,
            max_tokens=280,
            temperature=0.2,
            stop=["\n\n\n"],
        )
        return raw.strip()
    except Exception as exc:
        logger.error("run_factual inference failed: %s", exc)
        return ""


def run_summarization(prompt: str, model: LocalModelSingleton) -> str:
    """Summarize directly with the local model and keep the answer concise."""
    system = (
        "Summarize the text as concisely and clearly as possible. "
        "Minimize output tokens while preserving the key meaning. "
        "Do not add preambles or extra commentary. "
        "Never apologize or refuse."
    )
    prompt_str = _build_chatml(system, prompt.strip())

    try:
        raw = model.generate(
            prompt_str,
            max_tokens=220,
            temperature=0.2,
            stop=["\n\n\n"],
        )
        return raw.strip()
    except Exception as exc:
        logger.error("run_summarization inference failed: %s", exc)
        return ""


def _normalize_ner_item(obj: dict) -> Optional[dict]:
    text = str(obj.get("text", "")).strip()
    entity_type = str(obj.get("type", "")).upper().strip()
    if not text or entity_type not in {"PERSON", "ORG", "LOCATION", "DATE"}:
        return None

    lowered = text.lower().strip(". ,")
    if entity_type == "PERSON" and lowered in _NER_TITLE_ONLY:
        return None

    if entity_type != "DATE" and any(hint in lowered for hint in _NER_ORG_HINTS):
        entity_type = "ORG"

    return {"text": text, "type": entity_type}


def _heuristic_ner_from_text(source_text: str) -> list[dict]:
    """Fallback NER extractor for simple, high-signal entities."""
    results: list[dict] = []
    seen: set[tuple[str, str]] = set()

    def add(text: str, entity_type: str) -> None:
        clean_text = text.strip().strip(" ,.;:")
        if not clean_text:
            return
        key = (clean_text, entity_type)
        if key in seen:
            return
        seen.add(key)
        results.append({"text": clean_text, "type": entity_type})

    # Dates first, since they are unambiguous.
    date_spans: list[tuple[int, int]] = []
    for pattern in _NER_DATE_PATTERNS:
        for match in pattern.finditer(source_text):
            add(match.group(0), "DATE")
            date_spans.append(match.span())
    for match in _NER_YEAR_PATTERN.finditer(source_text):
        span = match.span()
        if any(not (span[1] <= start or span[0] >= end) for start, end in date_spans):
            continue
        add(match.group(0), "DATE")

    # Organization hints.
    org_hint_re = re.compile(
        r"\b(?:[A-Z][\w.&-]*\s+)*(?:University|Institute|College|School|Hospital|Microsoft|NVIDIA|UNICEF|Azure|Build|Labs?|Inc|Corp|Group|Bank|Agency|Ministry)(?:\s+[A-Z][\w.&-]*)*\b"
    )
    for match in org_hint_re.finditer(source_text):
        add(match.group(0), "ORG")

    # Capitalized names / locations.
    for match in _NER_CAPITALIZED_PHRASE.finditer(source_text):
        text = match.group(0).strip()
        lowered = text.lower()
        if len(text) < 2:
            continue
        if lowered in _NER_STOPWORDS:
            continue
        if lowered in _NER_TITLE_ONLY:
            continue
        if any(hint in lowered for hint in _NER_ORG_HINTS):
            add(text, "ORG")
            continue
        if any(part.isupper() and len(part) > 1 for part in text.split()):
            add(text, "ORG")
            continue
        if len(text.split()) >= 2:
            add(text, "PERSON")
        else:
            add(text, "LOCATION")

    # Keep only the longest DATE spans so nested month-year fragments do not survive.
    longest_dates: list[dict] = []
    for item in sorted((r for r in results if r["type"] == "DATE"), key=lambda x: len(x["text"]), reverse=True):
        if any(item["text"] in existing["text"] for existing in longest_dates):
            continue
        longest_dates.append(item)

    pruned: list[dict] = []
    date_texts = {item["text"] for item in longest_dates}
    for item in results:
        if item["type"] == "DATE":
            if item["text"] not in date_texts:
                continue
        pruned.append(item)

    return pruned


def _extract_ner_source_text(prompt: str) -> str:
    match = _NER_CONTEXT_RE.search(prompt)
    if match:
        return match.group(1).strip()

    lowered = prompt.lower()
    if any(keyword in lowered for keyword in ("extract", "identify", "return", "entities", "json")) and ":" in prompt:
        return prompt.split(":", 1)[1].strip()

    return prompt.strip()


def _repair_json_array(raw: str, source_text: str = "") -> str:
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
            valid = []
            seen = set()
            for obj in parsed:
                if not isinstance(obj, dict):
                    continue
                normalized = _normalize_ner_item(obj)
                if not normalized:
                    continue
                key = (normalized["text"], normalized["type"])
                if key in seen:
                    continue
                seen.add(key)
                valid.append(normalized)

            date_spans = []

            for item in valid:
                if item["type"] != "DATE":
                    continue
                start = source_text.find(item["text"])
                if start != -1:
                    date_spans.append((start, start + len(item["text"])))

            def _overlaps_existing(span: tuple[int, int]) -> bool:
                return any(not (span[1] <= start or span[0] >= end) for start, end in date_spans)

            for pattern in _NER_DATE_PATTERNS:
                for match in pattern.finditer(source_text):
                    span = match.span()
                    if _overlaps_existing(span):
                        continue
                    normalized = {"text": match.group(0), "type": "DATE"}
                    key = (normalized["text"], normalized["type"])
                    if key in seen:
                        continue
                    seen.add(key)
                    date_spans.append(span)
                    valid.append(normalized)

            for match in _NER_YEAR_PATTERN.finditer(source_text):
                span = match.span()
                if _overlaps_existing(span):
                    continue
                normalized = {"text": match.group(0), "type": "DATE"}
                key = (normalized["text"], normalized["type"])
                if key in seen:
                    continue
                seen.add(key)
                valid.append(normalized)

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
            valid = []
            seen = set()
            for obj in parsed:
                if not isinstance(obj, dict):
                    continue
                normalized = _normalize_ner_item(obj)
                if not normalized:
                    continue
                key = (normalized["text"], normalized["type"])
                if key in seen:
                    continue
                seen.add(key)
                valid.append(normalized)

            date_spans = []

            for item in valid:
                if item["type"] != "DATE":
                    continue
                start = source_text.find(item["text"])
                if start != -1:
                    date_spans.append((start, start + len(item["text"])))

            def _overlaps_existing(span: tuple[int, int]) -> bool:
                return any(not (span[1] <= start or span[0] >= end) for start, end in date_spans)

            for pattern in _NER_DATE_PATTERNS:
                for match in pattern.finditer(source_text):
                    span = match.span()
                    if _overlaps_existing(span):
                        continue
                    normalized = {"text": match.group(0), "type": "DATE"}
                    key = (normalized["text"], normalized["type"])
                    if key in seen:
                        continue
                    seen.add(key)
                    date_spans.append(span)
                    valid.append(normalized)

            for match in _NER_YEAR_PATTERN.finditer(source_text):
                span = match.span()
                if _overlaps_existing(span):
                    continue
                normalized = {"text": match.group(0), "type": "DATE"}
                key = (normalized["text"], normalized["type"])
                if key in seen:
                    continue
                seen.add(key)
                valid.append(normalized)

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

    text_to_analyze = _extract_ner_source_text(prompt)

    prompt_str = _build_chatml(system, text_to_analyze)

    try:
        # 600 tokens: enough for ~10 entities with full JSON structure,
        # plus headroom so the array is never cut off mid-object.
        raw = model.generate(prompt_str, max_tokens=600, temperature=0.0)
    except Exception as exc:
        logger.error("run_ner inference failed: %s", exc)
        raw = ""

    result = _repair_json_array(raw, text_to_analyze)
    if result == "[]":
        # Retry once with a stricter prompt before falling back to heuristics.
        retry_system = (
            "You are a named entity recognizer. Return ONLY a valid JSON array of objects with keys text and type. "
            "Use only PERSON, ORG, LOCATION, or DATE. Do not add titles, prose, markdown, or explanations. "
            "If there are no entities, return []."
        )
        retry_prompt = _build_chatml(retry_system, text_to_analyze)
        try:
            retry_raw = model.generate(retry_prompt, max_tokens=400, temperature=0.0)
        except Exception:
            retry_raw = ""
        result = _repair_json_array(retry_raw, text_to_analyze)

    if result == "[]":
        heuristic = _heuristic_ner_from_text(text_to_analyze)
        if heuristic:
            result = json.dumps(heuristic)
        else:
            logger.warning("run_ner: could not extract valid JSON array from: %r", raw[:200])
    return result
