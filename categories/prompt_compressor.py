"""
Prompt compressor — pure-Python, no ML dependencies.

Inspired by LLMLingua: scores sentences by TF-IDF importance and trims to a
token budget.  The first and last sentences are always preserved because they
typically contain the task instruction and closing constraint.

Compression is intentionally conservative: if the text is already short
enough the input is returned untouched, so callers do not need to gate on
length themselves.
"""
import math
import re
from collections import defaultdict


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Rough token count: words * 1.3, rounded to int.

    Good enough for logging / budget checks; not a BPE-accurate count.
    """
    return int(len(text.split()) * 1.3)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    """Split *text* into sentence-level chunks.

    Rules
    -----
    - Splits on ``.`` / ``!`` / ``?`` / newlines while keeping the delimiter
      attached to the preceding fragment.
    - Whole code blocks (any fragment that starts with `` ``` `` or contains
      `` ``` ``) are kept as a single unit — never split further.
    - Empty / whitespace-only fragments are filtered out.
    """
    # First, protect code blocks: extract them, replace with a unique
    # placeholder, re-insert after splitting so they are never broken.
    code_block_re = re.compile(r"```.*?```", re.DOTALL)
    placeholders: list[str] = []

    def _stash(m: re.Match) -> str:
        placeholders.append(m.group(0))
        return f"\x00CODEBLOCK{len(placeholders) - 1}\x00"

    protected = code_block_re.sub(_stash, text)

    # Split on sentence-ending punctuation or newlines.
    raw_parts = re.split(r"(?<=[.!?])\s+|\n+", protected)

    sentences: list[str] = []
    for part in raw_parts:
        part = part.strip()
        if not part:
            continue
        # Restore code-block placeholders.
        for idx, block in enumerate(placeholders):
            part = part.replace(f"\x00CODEBLOCK{idx}\x00", block)
        sentences.append(part)

    return sentences


def _tokenize(text: str) -> list[str]:
    """Lower-case word tokenizer that strips punctuation."""
    return re.findall(r"[a-z0-9']+", text.lower())


def _tfidf_scores(sentences: list[str]) -> list[float]:
    """Return a TF-IDF importance score for each sentence.

    Score(sentence) = mean TF-IDF over the words in that sentence.
    IDF uses log( N / df ) with N = number of sentences.
    """
    N = len(sentences)
    if N == 0:
        return []

    # Build per-sentence term frequency dicts.
    tf_per_sentence: list[dict[str, float]] = []
    df: dict[str, int] = defaultdict(int)

    for sent in sentences:
        words = _tokenize(sent)
        tf: dict[str, float] = defaultdict(float)
        total = max(1, len(words))
        for w in words:
            tf[w] += 1.0 / total
        tf_per_sentence.append(dict(tf))
        for w in set(words):
            df[w] += 1

    # Compute IDF.
    idf: dict[str, float] = {
        w: math.log(N / count) for w, count in df.items()
    }

    # Score each sentence.
    scores: list[float] = []
    for tf in tf_per_sentence:
        words_in_sent = list(tf.keys())
        if not words_in_sent:
            scores.append(0.0)
            continue
        score = sum(tf[w] * idf.get(w, 0.0) for w in words_in_sent)
        scores.append(score / max(1, len(words_in_sent)))

    return scores


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------

def compress(text: str, max_tokens: int = 300, ratio: float = 0.6) -> str:
    """Compress *text* to stay within *max_tokens* words using TF-IDF scoring.

    Parameters
    ----------
    text:
        The raw prompt text to compress.
    max_tokens:
        Word-count budget for the output (words, not BPE tokens).
    ratio:
        Unused by the scoring logic directly; kept for API compatibility and
        could be used by callers to derive ``max_tokens`` from the original
        length.

    Returns
    -------
    str
        The compressed text, or *text* unchanged if it was already short.
    """
    # Fast path: already short enough.
    if len(text.split()) <= max_tokens * 0.75:
        return text

    sentences = _split_sentences(text)
    if len(sentences) <= 2:
        # Nothing useful to trim.
        return text

    scores = _tfidf_scores(sentences)

    first_idx = 0
    last_idx = len(sentences) - 1
    middle_indices = list(range(1, last_idx))  # excludes first and last

    # Sort middle sentences by score descending, then greedily pick until
    # we hit the word budget (reserving room for first + last).
    first_words = len(sentences[first_idx].split())
    last_words = len(sentences[last_idx].split())
    budget = max_tokens - first_words - last_words

    ranked_middle = sorted(middle_indices, key=lambda i: scores[i], reverse=True)

    kept_middle: set[int] = set()
    used = 0
    for idx in ranked_middle:
        w = len(sentences[idx].split())
        if used + w > budget:
            continue  # skip this sentence; try a shorter one
        kept_middle.add(idx)
        used += w
        if used >= budget:
            break

    # Reassemble in original order.
    selected_indices = sorted({first_idx, last_idx} | kept_middle)
    parts = [sentences[i] for i in selected_indices]

    result = " ".join(parts)
    # Collapse runs of whitespace / blank lines.
    result = re.sub(r"[ \t]{2,}", " ", result)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def compress_messages(messages: list, max_tokens: int = 300) -> list:
    """Compress the 'content' field of every 'user' message in *messages*.

    System messages are passed through unchanged.  If no user message is
    present the original list is returned as-is.

    Parameters
    ----------
    messages:
        A list of ``{"role": str, "content": str}`` dicts (OpenAI chat format).
    max_tokens:
        Word-count budget forwarded to :func:`compress`.

    Returns
    -------
    list
        A new messages list; original dicts are not mutated.
    """
    has_user = any(m.get("role") == "user" for m in messages)
    if not has_user:
        return messages

    result: list = []
    for msg in messages:
        if msg.get("role") == "user":
            compressed_content = compress(msg["content"], max_tokens=max_tokens)
            result.append({**msg, "content": compressed_content})
        else:
            result.append(msg)
    return result
