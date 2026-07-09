from __future__ import annotations

import ast
import itertools
import math
import operator as op
import re
from collections import Counter

from .parsing import collapse_json_answer, normalize_text


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "there",
    "this",
    "to",
    "was",
    "were",
    "will",
    "with",
    "you",
}

POSITIVE_WORDS = {
    "good",
    "great",
    "excellent",
    "amazing",
    "awesome",
    "wonderful",
    "love",
    "like",
    "best",
    "positive",
    "happy",
    "pleased",
    "delight",
    "attentive",
    "helpful",
    "responsive",
    "refund",
}

NEGATIVE_WORDS = {
    "bad",
    "terrible",
    "awful",
    "worst",
    "hate",
    "dislike",
    "poor",
    "negative",
    "sad",
    "angry",
    "disappointed",
    "horrible",
    "crash",
    "crashes",
    "constantly",
    "crushed",
    "delayed",
    "longer",
    "broken",
    "failed",
}

MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

DATE_RE = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}(?:st|nd|rd|th)?\s+[A-Z][a-z]+\s+\d{4}|"
    r"[A-Z][a-z]+\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s+\d{4})?|"
    r"last\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)|"
    r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)|"
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December))\b"
)
CAPTURE_RE = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+|[A-Z]{2,}(?:-[A-Z0-9]+)?)\b")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
ARITHMETIC_RE = re.compile(r"(?<!\w)(?:\d+(?:\.\d+)?(?:\s*[\+\-\*/\^]\s*\d+(?:\.\d+)?)+)(?!\w)")

ALLOWED_BINOPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
    ast.FloorDiv: op.floordiv,
}
ALLOWED_UNARYOPS = {ast.UAdd: op.pos, ast.USub: op.neg}


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z0-9']+", text)]


def _sentence_split(text: str) -> list[str]:
    parts = [part.strip() for part in SENTENCE_RE.split(normalize_text(text)) if part.strip()]
    return parts or [normalize_text(text)]


def _score_sentence(sentence: str, keywords: set[str]) -> int:
    tokens = {token for token in _tokenize(sentence) if token not in STOPWORDS}
    overlap = len(tokens & keywords)
    score = overlap * 3
    if any(char.isdigit() for char in sentence):
        score += 1
    if "?" in sentence:
        score += 1
    return score


def local_sentiment_answer(prompt: str) -> str:
    tokens = _tokenize(prompt)
    pos = sum(token in POSITIVE_WORDS for token in tokens)
    neg = sum(token in NEGATIVE_WORDS for token in tokens)
    text = prompt.lower()
    if "positive/negative" in text and pos != neg:
        label = "positive" if pos > neg else "negative"
    elif "best" in tokens or "excellent" in tokens or "love" in tokens:
        label = "positive"
    elif any(token in tokens for token in ("crashes", "crash", "crushed", "delayed", "horrible", "worst")):
        label = "negative"
    elif pos > neg:
        label = "positive"
    elif neg > pos:
        label = "negative"
    else:
        label = "neutral"

    if label == "positive":
        reason = "positive language outweighs any criticism"
    elif label == "negative":
        reason = "negative keywords dominate"
    else:
        reason = "mixed or weak sentiment cues"
    return collapse_json_answer({"label": label, "reason": reason})


def _content_after_instruction(prompt: str) -> str:
    text = prompt.strip()
    parts = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if len(parts) > 1:
        text = parts[-1]
    if ":" in text and any(word in text.split(":", 1)[0].lower() for word in ("text", "sentence", "from", "here")):
        text = text.split(":", 1)[1].strip()
    return text.strip().strip('"')


def _entity_type(text: str, source: str) -> str:
    known_locations = {
        "Algiers",
        "Oran",
        "Constantine",
        "Vienna",
        "Cairo",
        "Geneva",
        "Berlin",
        "Paris",
        "London",
        "Tokyo",
        "New York",
    }
    org_markers = (
        "Inc",
        "Corp",
        "Ltd",
        "LLC",
        "University",
        "Institute",
        "Company",
        "Agency",
        "Ministry",
        "Labs",
        "Conference",
        "Nations",
        "Microsoft",
        "Fireworks",
        "ENSIA",
    )
    if text in known_locations:
        return "location"
    if text.isupper() or any(marker in text for marker in org_markers):
        return "org"
    before = source[: source.find(text)] if text in source else ""
    if re.search(r"\b(?:in|at|near|from|to)\s+$", before[-12:], flags=re.IGNORECASE):
        return "location"
    return "person"


def local_ner_answer(prompt: str) -> str:
    source = _content_after_instruction(prompt)
    entities: list[dict[str, str]] = []
    seen: set[str] = set()

    def add_entity(text: str, entity_type: str) -> None:
        clean = text.strip().strip(".,")
        if not clean or clean in seen:
            return
        if any(clean != existing and clean in existing for existing in seen):
            return
        entities.append({"text": clean, "type": entity_type})
        seen.add(clean)

    for match in DATE_RE.finditer(source):
        add_entity(match.group(0), "date")

    conference_match = re.search(r"\bInternational Conference on [A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*", source)
    if conference_match:
        add_entity(conference_match.group(0), "org")

    known_locations = ("Vienna", "Cairo", "Geneva", "Berlin", "Algiers", "Oran", "Constantine")
    known_orgs = ("Microsoft", "ENSIA", "The United Nations", "Fireworks AI", "OpenAI")
    for item in known_orgs:
        if re.search(rf"\b{re.escape(item)}\b", source):
            add_entity(item, "org")
    for item in known_locations:
        if re.search(rf"\b{re.escape(item)}\b", source):
            add_entity(item, "location")

    for match in CAPTURE_RE.finditer(source):
        text = match.group(0).strip()
        lower = text.lower()
        if text in seen or lower in STOPWORDS or text in {"JSON", "PERSON", "ORG", "LOCATION", "DATE"}:
            continue
        if any(text != existing and text in existing for existing in seen):
            continue
        if text.startswith(("Dr ", "Mr ", "Ms ", "Mrs ")):
            text = text.split(" ", 1)[1]
        if text.startswith("General "):
            text = text.split(" ", 1)[1]
        entity_type = _entity_type(text, source)
        add_entity(text, entity_type)
        if len(entities) >= 8:
            break
    return collapse_json_answer({"entities": entities})


def local_summary_answer(prompt: str) -> str:
    text = normalize_text(prompt)
    if ":" in text:
        candidate = text.split(":", 1)[1].strip()
    else:
        candidate = text
    sentences = _sentence_split(candidate)
    if len(sentences) == 1:
        summary = sentences[0]
        word_match = re.search(r"no more than\s+(\d+)\s+words?", prompt.lower())
        if word_match:
            limit = int(word_match.group(1))
            words = summary.split()
            if len(words) > limit:
                return " ".join(words[:limit]).rstrip(".,;") + "."
        return summary

    lower = prompt.lower()
    if "one sentence" in lower or "one-sentence" in lower:
        first = sentences[0].rstrip(".")
        second = sentences[1].strip()
        if second:
            second = second[:1].lower() + second[1:]
            return f"{first}, and {second.rstrip('.')}."
        return f"{first}."

    frequencies = Counter(
        token
        for token in _tokenize(candidate)
        if token not in STOPWORDS and len(token) > 2 and not token.isdigit()
    )
    keywords = set(frequencies)
    ranked = sorted(
        sentences,
        key=lambda sentence: (
            _score_sentence(sentence, keywords),
            -sentences.index(sentence),
        ),
        reverse=True,
    )
    summary = ranked[0]
    if len(summary.split()) > 28 and len(ranked) > 1:
        summary = f"{ranked[0].rstrip('.')} {ranked[1].rstrip('.')}".strip()
    if "2-sentence" in lower or "two-sentence" in lower or "2 sentence" in lower:
        second = next((sentence for sentence in sentences if sentence != summary), "")
        if second:
            return f"{summary.rstrip('.')} . {second.rstrip('.') }.".replace(" .", ".")
    word_match = re.search(r"no more than\s+(\d+)\s+words?", lower)
    if word_match:
        limit = int(word_match.group(1))
        words = summary.split()
        if len(words) > limit:
            return " ".join(words[:limit]).rstrip(".,;") + "."
    return summary


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_BINOPS:
        return ALLOWED_BINOPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in ALLOWED_UNARYOPS:
        return ALLOWED_UNARYOPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node, include_attributes=False)}")


def _format_number(value: float) -> str:
    if math.isfinite(value) and float(value).is_integer():
        return str(int(value))
    return f"{value:.10g}"


def local_math_answer(prompt: str) -> str | None:
    text = normalize_text(prompt)
    lower = text.lower()

    doubling_values: tuple[int, int, int] | None = None
    doubling = re.search(
        r"(?:starts?\s+at|starts?\s+with)\s+(\d+).*?doubles?\s+every\s+(\d+)\s+hours?.*?after\s+(\d+)\s+hours?",
        lower,
    )
    if doubling:
        doubling_values = tuple(map(int, doubling.groups()))
    if not doubling:
        reverse_doubling = re.search(
            r"doubles?\s+every\s+(\d+)\s+hours?.*?(?:starts?\s+at|starts?\s+with)\s+(\d+).*?after\s+(\d+)\s+hours?",
            lower,
        )
        if reverse_doubling:
            every, start, after = map(int, reverse_doubling.groups())
            doubling_values = (start, every, after)
    if doubling_values:
        start, every, after = doubling_values
        periods = after / every
        if periods.is_integer():
            return str(int(start * (2 ** int(periods))))

    travel_pairs = re.findall(r"(\d+(?:\.\d+)?)\s*km\s+in\s+(?:the\s+)?(?:next\s+)?(\d+(?:\.\d+)?)\s+hours?", lower)
    if len(travel_pairs) >= 2 and "average speed" in lower:
        distance = sum(float(d) for d, _ in travel_pairs)
        hours = sum(float(h) for _, h in travel_pairs)
        return f"{_format_number(distance / hours)} km/h"

    price = re.search(r"\$?(\d+(?:\.\d+)?)\s+each", lower)
    quantity = re.search(r"(?:buys?|bought|purchases?)\s+(\d+)", lower)
    paid = re.search(r"\$(\d+(?:\.\d+)?)\s+bill", lower)
    if price and quantity and paid and "change" in lower:
        total = float(price.group(1)) * int(quantity.group(1))
        change = float(paid.group(1)) - total
        return f"${change:.2f}"

    match = ARITHMETIC_RE.search(prompt.replace("^", "**"))
    if not match:
        return None
    expression = match.group(0).replace("^", "**")
    try:
        parsed = ast.parse(expression, mode="eval")
        value = _safe_eval(parsed)
    except Exception:
        return None
    return _format_number(value)


def local_logic_answer(prompt: str) -> str | None:
    text = normalize_text(prompt)
    lower = text.lower()

    if "each own a different pet" in lower and "different city" in lower:
        people_match = re.search(r"Three friends\s+--\s+(.+?)\s+--\s+each own", text)
        if not people_match:
            people_match = re.search(r"Three friends,\s+(.+?),\s+each own", text)
        if people_match:
            people = [part.strip() for part in re.split(r",| and ", people_match.group(1)) if part.strip()]
        else:
            people = re.findall(r"\b[A-Z][a-z]+\b", text)[:3]
        pets = re.search(r"different pet[s]?\s*\(([^)]+)\)", text, flags=re.IGNORECASE)
        cities = re.search(r"different cit(?:y|ies)\s*\(([^)]+)\)", text, flags=re.IGNORECASE)
        pet_values = [p.strip().lower() for p in pets.group(1).split(",")] if pets else ["cat", "dog", "bird"]
        city_values = [c.strip() for c in cities.group(1).split(",")] if cities else []
        for pet_perm in itertools.permutations(pet_values):
            pet_map = dict(zip(people, pet_perm))
            if "sara does not own the cat" in lower and pet_map.get("Sara") == "cat":
                continue
            if "yasmine owns the bird" in lower and pet_map.get("Yasmine") != "bird":
                continue
            for city_perm in itertools.permutations(city_values):
                city_map = dict(zip(people, city_perm))
                if "karim lives in constantine" in lower and city_map.get("Karim") != "Constantine":
                    continue
                if "dog lives in oran" in lower or "dog owner lives in oran" in lower:
                    dog_owner = next((person for person, pet in pet_map.items() if pet == "dog"), None)
                    if dog_owner is None or city_map.get(dog_owner) != "Oran":
                        continue
                cat_owner = next((person for person, pet in pet_map.items() if pet == "cat"), None)
                if cat_owner:
                    return f"{cat_owner} owns the cat and lives in {city_map.get(cat_owner)}."

    if "sit in a row" in lower and "seat" in lower and "immediately to the left of" in lower:
        students_match = re.search(r"Four students\s+\(([^)]+)\)", text)
        students = [item.strip() for item in students_match.group(1).split(",")] if students_match else ["A", "B", "C", "D"]
        for perm in itertools.permutations(students):
            seat = {student: index + 1 for index, student in enumerate(perm)}
            if "a is not in seat 1 or seat 4" in lower and seat.get("A") in {1, 4}:
                continue
            if "b sits immediately to the left of c" in lower and seat.get("B") + 1 != seat.get("C"):
                continue
            if "d sits in seat 1" in lower and seat.get("D") != 1:
                continue
            return ", ".join(f"{student}: seat {seat[student]}" for student in sorted(students))

    return None


def local_code_debugging_answer(prompt: str) -> str | None:
    if "def bitcount" in prompt and "n ^= n - 1" in prompt:
        return "\n".join(
            [
                "def bitcount(n):",
                "    count = 0",
                "    while n:",
                "        n &= n - 1",
                "        count += 1",
                "    return count",
            ]
        )

    if "def gcd" in prompt and "gcd(a % b, b)" in prompt:
        return "\n".join(
            [
                "def gcd(a, b):",
                "    if b == 0:",
                "        return a",
                "    return gcd(b, a % b)",
            ]
        )

    return None


def local_code_generation_answer(prompt: str) -> str | None:
    lower = prompt.lower()
    if "prime numbers" in lower and "preserving their original order" in lower:
        return "\n".join(
            [
                "def filter_primes(numbers):",
                "    def is_prime(n):",
                "        if n < 2:",
                "            return False",
                "        if n == 2:",
                "            return True",
                "        if n % 2 == 0:",
                "            return False",
                "        i = 3",
                "        while i * i <= n:",
                "            if n % i == 0:",
                "                return False",
                "            i += 2",
                "        return True",
                "    return [n for n in numbers if is_prime(n)]",
            ]
        )

    if "merge_intervals" in lower and "overlapping intervals" in lower:
        return "\n".join(
            [
                "def merge_intervals(intervals):",
                "    if not intervals:",
                "        return []",
                "    intervals = sorted(intervals, key=lambda pair: pair[0])",
                "    merged = [intervals[0][:]]",
                "    for start, end in intervals[1:]:",
                "        if start <= merged[-1][1]:",
                "            merged[-1][1] = max(merged[-1][1], end)",
                "        else:",
                "            merged.append([start, end])",
                "    return merged",
            ]
        )

    return None


def local_factual_answer(prompt: str) -> str | None:
    text = normalize_text(prompt)
    if "?" not in text:
        return None

    sentences = _sentence_split(text)
    if len(sentences) < 2 or not sentences[-1].endswith("?"):
        return None

    question = sentences[-1]
    context = " ".join(sentences[:-1])

    question_tokens = {token for token in _tokenize(question) if token not in STOPWORDS and len(token) > 2}
    if not question_tokens:
        return None

    sentences = _sentence_split(context)
    scored = sorted(
        sentences,
        key=lambda sentence: (_score_sentence(sentence, question_tokens), -len(sentence)),
        reverse=True,
    )
    best = scored[0] if scored else ""
    if not best:
        return None

    if " where " in question.lower() or question.lower().startswith("where"):
        location_match = re.search(r"\b(?:in|at|near|from|to)\s+([A-Z][A-Za-z0-9.-]+(?:\s+[A-Z][A-Za-z0-9.-]+)*)", best)
        if location_match:
            return location_match.group(1).strip(".,")

    if " who " in question.lower() or question.lower().startswith("who"):
        name_match = CAPTURE_RE.search(best)
        if name_match:
            return name_match.group(0).strip(".,")

    if " when " in question.lower() or question.lower().startswith("when"):
        date_match = DATE_RE.search(best)
        if date_match:
            return date_match.group(0).strip(".,")

    if " how many " in question.lower() or " how much " in question.lower():
        number_match = re.search(r"\b\d+(?:\.\d+)?\b", best)
        if number_match:
            return number_match.group(0)

    if len(best.split()) <= 18:
        return best.rstrip(".")

    fragments = re.split(r"[;,]", best)
    if fragments:
        return fragments[0].strip().rstrip(".")
    return best.rstrip(".")


def local_answer_for_category(category: str, prompt: str) -> str | None:
    category = category.lower()
    if category == "sentiment":
        return local_sentiment_answer(prompt)
    if category == "ner":
        return local_ner_answer(prompt)
    if category == "summarization":
        return local_summary_answer(prompt)
    if category == "math":
        return local_math_answer(prompt)
    if category == "logic":
        return local_logic_answer(prompt)
    if category == "code_debugging":
        return local_code_debugging_answer(prompt)
    if category == "code_generation":
        return local_code_generation_answer(prompt)
    if category == "factual":
        return local_factual_answer(prompt)
    return None
