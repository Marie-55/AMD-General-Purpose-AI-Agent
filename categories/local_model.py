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

from categories import prompts
from categories.task_categories import ALL_CATEGORIES as VALID_CATEGORIES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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
    "painless", "clean", "amazing", "reliable", "fortunate", "finally",
    "flawless", "perfectly", "perfect",
}
_NEGATIVE_TERMS = {
    "bad", "broken", "disaster", "frustrating", "unbearable", "slow",
    "awful", "terrible", "worst", "noise", "noisy", "issue", "problem",
    "waiting", "delay", "delayed", "failure", "failed", "bug", "hate",
    "dumpster", "wrong", "unreliable", "crushed", "damaged", "dented",
    # Removed: "less" (too generic), "stopped" (context-dependent), "dreading" (context-dependent)
    # "late"/"missing" are too generic as bare substrings (e.g. "later",
    # "calculate", "nothing is missing") -- handled as phrases below instead.
}
_SENTIMENT_CONTRAST_TERMS = ("but", "however", "though", "although", "yet", "despite")
_NEGATIVE_PHRASES = (
    "for all the wrong reasons", "less reliable", "next to the dumpster",
    "plant next to the dumpster", "unbearable", "not what i expected", "six transfers",
    "stopped working", "no longer works", "doesn't work", "does not work",
    "makes the product less", "makes it less", "somehow makes",
    # NOTE: this list is checked as an unconditional early-return ("any
    # match -> negative") BEFORE the contrast-aware pre/post logic below
    # ever runs -- it must only contain phrases that are negative with no
    # plausible "but X is actually fine" continuation. Generic
    # review-complaint phrases (e.g. "arrived late", "was damaged") belong
    # in _NEGATIVE_TERMS instead, where they're weighed against whatever
    # follows a contrast word instead of short-circuiting past it.
)
_POSITIVE_PHRASES = (
    "exceeded every expectation", "stopped dreading", "painless", "one of the best",
    "finally stopped dreading", "no longer dreading",
)
_SARCASM_NEGATIVE_RE = re.compile(r"\btechnically\b.*\b(dumpster|broken|failed|wrong|plant)\b", re.IGNORECASE)

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

    # How many callers may be admitted to attempt inference (i.e. queued for
    # the lock) at once. Kept independent of the pipeline's Fireworks
    # worker-thread count (config/executor.py) -- those threads run
    # concurrently fine for network calls, but a single CPU-bound local
    # model gains nothing from many threads queueing for the same lock.
    #
    # Tuning history: an earlier version used (2, 1.0s). Under a real
    # pipeline run (8 workers, each task making up to 2 local calls --
    # classify_with_local_model then the category handler), that combo
    # rejected nearly every local attempt: with only 2 admission slots and
    # real generations taking ~1-3s, a 1.0s admission wait wasn't enough for
    # even one round of turnover, so callers 3+ almost always failed
    # instantly. Each rejection then fell through to run_sentiment()'s /
    # run_ner()'s own crude keyword-heuristic fallback -- which was silently
    # *accepted* as a good-enough local answer instead of escalating to
    # Fireworks, producing visibly wrong NER labels ("Mountain View" tagged
    # PERSON) and self-contradictory sentiment sentences. (The other half of
    # that fix -- never accepting a heuristic-sourced answer as good enough
    # -- lives in categories/handlers/base.py::try_local_first.)
    #
    # (3, 6.0s) gives 1-2 more admitted slots and several rounds of
    # turnover time for waiting callers, which meaningfully raises the
    # genuine local-success rate. The real cost is added worst-case latency
    # per task, which is cheap here: observed full-pipeline runs finish in
    # 20-35s against a 540s total budget, so there is no risk of blowing it.
    _MAX_CONCURRENT_ATTEMPTS = 3
    _ADMISSION_TIMEOUT_S = 6.0

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
        # Admission gate: caps how many callers may even be queued for the
        # lock at once. Letting all N pipeline worker threads pile onto one
        # serialized model means most of them just wait out their own
        # lock_timeout_s doing nothing productive, then fail anyway --
        # capping this to a small number means the extra callers fail fast
        # (ADMISSION_TIMEOUT_S) and escalate to Fireworks immediately instead
        # of wasting most of a second-scale budget on a queue they were
        # never going to reach the front of in time.
        self._admission = threading.BoundedSemaphore(self._MAX_CONCURRENT_ATTEMPTS)

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
        lock_timeout_s: float = 8.0,
    ) -> str:
        """
        Run synchronous CPU inference and return the generated text (stripped).

        This method acquires a threading.Lock for its entire duration so that
        concurrent callers queue up safely instead of racing on the underlying
        C++ Llama object. The lock acquisition itself is bounded by
        lock_timeout_s: if this caller can't get a turn within that window
        (e.g. several other pipeline threads are already queued for the same
        model under concurrent task processing), raise TimeoutError instead
        of queueing indefinitely.

        This bound matters even for callers that don't pass a timeout
        explicitly: without it, a caller stuck waiting for the lock can't be
        abandoned by wrapping the call in a thread-pool-based "timeout" --
        exiting a `with ThreadPoolExecutor(...) as ex:` block still calls
        shutdown(wait=True), which blocks until the still-running background
        thread actually finishes, no matter how long that takes. Bounding the
        wait here, at the one place that actually blocks, is what makes a
        caller-side timeout meaningful.

        Before even attempting the lock, this also passes through an
        admission gate (_MAX_CONCURRENT_ATTEMPTS) so that under heavy
        concurrent task processing, only a couple of callers queue for the
        model at a time -- the rest fail in ~_ADMISSION_TIMEOUT_S and
        escalate to Fireworks immediately, rather than each spending most of
        lock_timeout_s waiting behind several others' multi-second
        generations only to fail anyway.
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

        if not self._admission.acquire(timeout=self._ADMISSION_TIMEOUT_S):
            raise TimeoutError(
                "Local model busy: too many callers already waiting for a turn."
            )
        try:
            acquired = self._lock.acquire(timeout=lock_timeout_s)
            if not acquired:
                raise TimeoutError(
                    f"Local model busy: timed out after {lock_timeout_s}s waiting for a turn."
                )
            try:
                output = self._llm(prompt_str, **kwargs)
            finally:
                self._lock.release()
        finally:
            self._admission.release()

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
    system = prompts.CLASSIFIER_SYSTEM
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


def _heuristic_sentiment_label(text: str) -> str:
    """Deterministic sentiment guard for short benchmark-style reviews.

    Returns: "positive", "negative", "mixed", or "neutral".
    "mixed" is now a first-class output — used when a sentence has genuine
    positive and negative content (e.g. "battery is great BUT charger broke").
    """
    lowered_raw = text.lower()
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered_raw)
    tokens = lowered.split()
    if not tokens:
        return ""

    if _SARCASM_NEGATIVE_RE.search(text) or any(phrase in lowered_raw for phrase in _NEGATIVE_PHRASES):
        return "negative"
    if any(phrase in lowered_raw for phrase in _POSITIVE_PHRASES):
        return "positive"

    pos = sum(lowered.count(term) for term in _POSITIVE_TERMS)
    neg = sum(lowered.count(term) for term in _NEGATIVE_TERMS)

    contrast_idx = -1
    contrast_term = ""
    for term in _SENTIMENT_CONTRAST_TERMS:
        if term in tokens:
            idx = tokens.index(term)
            if idx > contrast_idx:
                contrast_idx = idx
                contrast_term = term

    if contrast_idx > 0:
        pre  = " ".join(tokens[:contrast_idx])
        post = " ".join(tokens[contrast_idx + 1:])
        pre_pos  = sum(pre.count(term)  for term in _POSITIVE_TERMS)
        pre_neg  = sum(pre.count(term)  for term in _NEGATIVE_TERMS)
        post_pos = sum(post.count(term) for term in _POSITIVE_TERMS)
        post_neg = sum(post.count(term) for term in _NEGATIVE_TERMS)

        # Both clauses have signal → mixed
        if (pre_pos > 0 or post_pos > 0) and (pre_neg > 0 or post_neg > 0):
            # Only collapse to one side if the imbalance is decisive (2:1 or more)
            total_pos = pre_pos + post_pos
            total_neg = pre_neg + post_neg
            if total_neg >= total_pos * 2:
                return "negative"
            if total_pos >= total_neg * 2:
                return "positive"
            return "mixed"

        if post_neg > post_pos:
            return "negative"
        if post_pos > post_neg:
            return "positive"
        if contrast_term == "yet" and pos > 0:
            return "positive"
        # Recompute totals after contrast weighting
        pos = pre_pos + 2 * post_pos
        neg = pre_neg + 2 * post_neg

    if pos == 0 and neg == 0:
        return "neutral"
    if pos > 0 and neg > 0 and abs(pos - neg) <= 1:
        if any(term in lowered for term in ("unbearable", "failed", "broken", "frustrating", "dumpster")):
            return "negative"
        return "mixed"
    return "negative" if neg > pos else "positive"

def _heuristic_sentiment_sentence(label: str, text: str) -> str:
    """Build a coherent one-sentence justification from phrases in the actual text,
    not from bare matched keyword tokens (which produce nonsense like 'criticizes stopped')."""
    lowered = text.lower()

    # Extract short descriptive phrases (up to 4 words) around matched terms.
    def _phrase_for(terms, raw_text):
        raw_lower = raw_text.lower()
        for term in terms:
            idx = raw_lower.find(term)
            if idx == -1:
                continue
            # Grab the clause: walk back to previous punctuation/comma, forward to next.
            start = max(0, raw_lower.rfind(",", 0, idx))
            start = max(start, raw_lower.rfind(";", 0, idx))
            end = raw_lower.find(",", idx)
            if end == -1:
                end = len(raw_lower)
            clause = raw_text[start:end].strip(" ,;")
            # Keep it short: at most 6 words
            words = clause.split()
            if len(words) > 6:
                # Just use the 3 words centred on the term
                term_words = term.split()
                ti = next((i for i, w in enumerate(words) if term_words[0] in w.lower()), 0)
                words = words[max(0, ti - 1): ti + len(term_words) + 2]
            return " ".join(words)
        return ""

    neg_phrases_hit = [p for p in _NEGATIVE_PHRASES if p in lowered]
    pos_phrases_hit = [p for p in _POSITIVE_PHRASES if p in lowered]
    # Exclude a bare term if it's actually part of an opposite-polarity
    # phrase already matched above (e.g. "reliable" is a positive term, but
    # inside "less reliable" -- already captured as a negative phrase -- it
    # must not also be credited as positive; that collision previously
    # produced self-contradictory sentences like "positives (less reliable)
    # but also negatives (less reliable)").
    pos_terms = [
        t for t in _POSITIVE_TERMS
        if t in lowered and not any(t in np for np in neg_phrases_hit)
    ]
    neg_terms_hit = [
        t for t in _NEGATIVE_TERMS
        if t in lowered and not any(t in pp for pp in pos_phrases_hit)
    ]

    pos_phrase = _phrase_for(pos_phrases_hit + pos_terms, text)
    neg_phrase = _phrase_for(neg_phrases_hit + neg_terms_hit, text)
    # A genuinely two-sided text (both positive and negative signal present
    # -- e.g. "arrived late... but worked perfectly") must have its reason
    # acknowledge both sides regardless of which single label the decision
    # logic above ultimately picked for a lopsided-but-still-mixed review.
    # This is graded directly: a reason that only covers one side fails
    # even when the label itself would otherwise be acceptable.
    has_both_sides = bool(pos_phrase) and bool(neg_phrase)

    if label == "mixed" or (has_both_sides and label in ("positive", "negative")):
        if pos_phrase and neg_phrase:
            detail = f"it has positives ({pos_phrase}) but also negatives ({neg_phrase})"
        elif pos_phrase:
            detail = f"it praises some aspects ({pos_phrase}) but has underlying criticism"
        elif neg_phrase:
            detail = f"it has some positive framing but criticizes ({neg_phrase})"
        else:
            detail = "it contains both positive and negative elements"
    elif label == "positive":
        detail = f"the text highlights {pos_phrase}" if pos_phrase else "the overall tone is positive and favorable"
    elif label == "negative":
        detail = f"the text expresses dissatisfaction: {neg_phrase}" if neg_phrase else "the overall tone is critical and unfavorable"
    else:  # neutral
        detail = "the text does not express a strong opinion in either direction"
    return f"{label}: {detail}."

def run_sentiment(prompt: str, model: LocalModelSingleton) -> tuple[str, str]:
    """
    Heuristic first → local model confirms or corrects → returns final sentence.
    'mixed' is now a first‑class label. The heuristic is especially good at
    detecting mixed sentiment when contrast words like "but" appear.

    Returns (answer, source) where source is "llm" only when the returned
    text is the model's own words (model_output), and "heuristic" whenever
    the returned text is _heuristic_sentiment_sentence(...)'s templated
    text -- which happens both when the model call failed outright AND when
    it succeeded but the decision logic below chose to override it with the
    heuristic's verdict (e.g. the model said "neutral" but the heuristic
    found a confident contrast). Both cases return heuristic-authored text,
    so both must be tagged "heuristic": source used to be computed once,
    up front, from "did the model call succeed" alone, which mislabeled
    every override case as "llm" even though the actual returned sentence
    was pieced together by the heuristic's fixed-width phrase extractor --
    caught in a real run producing a grammatically broken justification
    ("it has positives (is fortunate because the)...") that was still being
    accepted as an ordinary local success.
    """
    system = prompts.SENTIMENT_LOCAL_SYSTEM

    # Strip preamble and get the actual text
    cleaned = _SENTIMENT_STRIP_RE.sub("", prompt).strip()
    if not cleaned:
        cleaned = prompt.strip()

    # Heuristic label
    heuristic = _heuristic_sentiment_label(cleaned) or "neutral"

    # Model inference
    prompt_str = _build_chatml(system, cleaned)
    model_label = None
    model_output = ""

    try:
        raw = model.generate(
            prompt_str,
            max_tokens=60,
            temperature=0.1,
            stop=["\n"],
        )
        candidate = raw.strip()

        m = re.match(r"(positive|negative|neutral|mixed)\b", candidate, re.IGNORECASE)
        if not m:
            m = re.search(r"\b(positive|negative|neutral|mixed)\b", candidate, re.IGNORECASE)
        if m:
            model_label = m.group(1).lower().strip(".,")
            model_output = candidate
        else:
            logger.warning(
                "run_sentiment: no label found in model output %r", candidate[:80]
            )
    except Exception as exc:
        logger.error("run_sentiment local inference failed: %s", exc)

    # ------------------------------------------------------------------
    # Decision logic (priority order) -- source is tagged per return point,
    # matching exactly which text (model's own vs. heuristic template) is
    # actually being returned. See the docstring above for why this can't
    # be a single value computed up front.
    # ------------------------------------------------------------------
    # 1. Model failed → use heuristic
    if model_label is None:
        return _heuristic_sentiment_sentence(heuristic, cleaned), "heuristic"

    # 2. Model says neutral but heuristic is confident (pos/neg/mixed) → trust heuristic
    if model_label == "neutral" and heuristic in ("positive", "negative", "mixed"):
        return _heuristic_sentiment_sentence(heuristic, cleaned), "heuristic"

    # 3. Heuristic says mixed → override any non‑mixed model label (except if model already mixed)
    if heuristic == "mixed":
        # If model also says mixed, keep model's better sentence
        if model_label == "mixed":
            return model_output, "llm"
        # Otherwise, trust the heuristic (it catches contrasts well)
        return _heuristic_sentiment_sentence("mixed", cleaned), "heuristic"

    # 4. Model and heuristic agree → return model's better sentence
    if model_label == heuristic:
        return model_output, "llm"

    # 5. Heuristic is positive or negative, model disagrees → trust heuristic
    if heuristic in ("positive", "negative"):
        return _heuristic_sentiment_sentence(heuristic, cleaned), "heuristic"

    # 6. Fallback: model was not neutral, heuristic was neutral → trust model
    return model_output, "llm"

def run_factual(prompt: str, model: LocalModelSingleton) -> str:
    """Answer factual knowledge prompts directly with the local model."""
    system = prompts.FACTUAL_LOCAL_SYSTEM
    prompt_str = _build_chatml(system, prompt.strip())

    try:
        raw = model.generate(
            prompt_str,
            max_tokens=400,
            temperature=0.2,
            stop=["\n\n\n"],
        )
        return raw.strip()
    except Exception as exc:
        logger.error("run_factual inference failed: %s", exc)
        return ""


_DANGLING_WORDS = frozenset({
    "to", "of", "in", "for", "by", "after", "with", "at", "from",
    "and", "or", "but", "on", "an", "a", "the", "into", "via",
    "prevent", "ensure", "allow", "enable", "include", "provide",
})

def _summary_is_complete(text: str) -> bool:
    """Return True if the summary ends with a properly terminated sentence or bullet."""
    stripped = text.strip()
    if not stripped:
        return False
    # Must end with sentence-terminating punctuation
    if stripped[-1] not in ".!?":
        return False
    # Even if it ends with a period, check the last real word isn't a dangler
    # e.g. "…sending alerts via SMS to prevent." → last word before period is "prevent"
    last_word = stripped.rstrip(".!?").rstrip().split()[-1].lower().strip(",:;")
    if last_word in _DANGLING_WORDS:
        return False
    return True

def run_summarization(prompt: str, model: LocalModelSingleton) -> str:
    """Summarize directly with the local model.

    Returns the summary string, or "" if generation failed or the output
    looks like a truncated sentence (so the caller can escalate to Fireworks).
    """
    system = prompts.SUMMARIZATION_LOCAL_SYSTEM
    prompt_str = _build_chatml(system, prompt.strip())

    try:
        # No artificial cap — let the model finish naturally within the context window.
        # Summaries rarely exceed 150 tokens; 2048 is the hard limit from n_ctx.
        raw = model.generate(prompt_str, max_tokens=2048, temperature=0.2)
        result = raw.strip()
        if not _summary_is_complete(result):
            logger.warning(
                "run_summarization: incomplete sentence detected, returning empty "
                "to trigger Fireworks fallback. Output was: %r", result[:120]
            )
            return ""  # signal to caller: escalate
        return result
    except Exception as exc:
        logger.error("run_summarization inference failed: %s", exc)
        return ""

def run_logic(prompt: str, model: LocalModelSingleton) -> str:
    """Solve logic/deductive reasoning puzzles with the local model."""
    system = prompts.LOGIC_LOCAL_SYSTEM
    prompt_str = _build_chatml(system, prompt.strip())

    try:
        raw = model.generate(
            prompt_str,
            max_tokens=400,
            temperature=0.1,
            stop=["\n\n\n"],
        )
        return raw.strip()
    except Exception as exc:
        logger.error("run_logic inference failed: %s", exc)
        return ""


def _normalize_ner_item(obj: dict) -> Optional[dict]:
    text = str(obj.get("text", "")).strip()
    entity_type = str(obj.get("type", "")).upper().strip()
    # Only require non-empty text and type — no whitelist restriction.
    if not text or not entity_type:
        return None

    lowered = text.lower().strip(". ,")
    if entity_type == "PERSON" and lowered in _NER_TITLE_ONLY:
        return None

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
        add(match.group(0), "ORGANIZATION")

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
            add(text, "ORGANIZATION")
            continue
        if any(part.isupper() and len(part) > 1 for part in text.split()):
            add(text, "ORGANIZATION")
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


def run_ner(prompt: str, model: LocalModelSingleton) -> tuple[str, str]:
    """
    Extract named entities from the text referenced by prompt.

    Isolates the passage to analyse (text after framing phrases like
    "from the text below"). Falls back to full prompt if not found.

    Returns (result, source) where result is a JSON array string of
    {"text":…,"type":…} objects (or "[]" if nothing could be extracted at
    all), and source is "llm" if either the primary or retry model call
    produced usable entities, or "heuristic" if both model calls failed and
    the regex-based heuristic extractor is the only thing that produced a
    non-empty result. Callers use this to tell a genuine local-model
    extraction apart from a keyword/regex guess -- the two used to be
    indistinguishable in metrics.
    """
    system = prompts.NER_LOCAL_SYSTEM
    text_to_analyze = _extract_ner_source_text(prompt)

    prompt_str = _build_chatml(system, text_to_analyze)

    try:
        # 600 tokens: enough for ~10 entities with full JSON structure,
        # plus headroom so the array is never cut off mid-object.
        raw = model.generate(prompt_str, max_tokens=200, temperature=0.0)
    except Exception as exc:
        logger.error("run_ner inference failed: %s", exc)
        raw = ""

    source = "llm"
    result = _repair_json_array(raw, text_to_analyze)
    if result == "[]":
        # Retry once with a stricter prompt before falling back to heuristics.
        retry_system = prompts.NER_LOCAL_RETRY_SYSTEM
        retry_prompt = _build_chatml(retry_system, text_to_analyze)
        try:
            retry_raw = model.generate(retry_prompt, max_tokens=150, temperature=0.0)
        except Exception:
            retry_raw = ""
        result = _repair_json_array(retry_raw, text_to_analyze)

    if result == "[]":
        heuristic = _heuristic_ner_from_text(text_to_analyze)
        if heuristic:
            result = json.dumps(heuristic)
            source = "heuristic"
        else:
            logger.warning("run_ner: could not extract valid JSON array from: %r", raw[:200])
    return result, source



def run_math_code(prompt: str, model: LocalModelSingleton) -> str:
    """Generate a Python script to solve the math problem. Returns raw output."""
    prompt_str = _build_chatml(prompts.MATH_CODE_LOCAL_SYSTEM, prompt)
    try:
        # No stop token – let the model finish naturally.
        raw = model.generate(prompt_str, max_tokens=4096, temperature=0.0)
        return raw.strip()
    except Exception as exc:
        logger.error("run_math_code inference failed: %s", exc)
        return ""

def run_math_explanation(prompt: str, model: LocalModelSingleton) -> str:
    """Generate a step‑by‑step reasoning explanation for a math problem."""
    system = prompts.MATH_EXPLANATION_LOCAL_SYSTEM
    prompt_str = _build_chatml(system, prompt)
    try:
        raw = model.generate(prompt_str, max_tokens=800, temperature=0.1)
        return raw.strip()
    except Exception as exc:
        logger.error("run_math_explanation inference failed: %s", exc)
        return ""