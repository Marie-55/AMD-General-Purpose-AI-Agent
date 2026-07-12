"""
Self-check against tests/judge_guide.md's public validation examples.

Not a pytest test (deliberately not named test_*.py, matching the
convention in check_model_availability.py) -- this checks OUTPUT CONTENT
against the rubric text in judge_guide.md, not code. It approximates the
hidden LLM judge with hardcoded structural/keyword checks per task_id, since
we don't have a real judge to call locally. A PASS here is a strong signal;
a FAIL is close to certain to also fail the real judge (the checks encode
things judge_guide.md states explicitly, e.g. "must make the subset
relationship clear", "does not pass" for missing a required point). It is
NOT a guarantee of a real pass -- only the actual hidden judge decides that.

Usage:
    python3 tests/check_judge_samples.py path/to/results.json

Run the judge sample tasks first, e.g.:
    set -a && source .env && set +a
    python3 tests/run_fireworks_pipeline.py \\
        --input input/tasks_judge_samples.json --output output/judge_samples_results.json
    python3 tests/check_judge_samples.py output/judge_samples_results.json
"""
import json
import re
import sys


def _contains_any(text: str, needles: list[str]) -> bool:
    lowered = text.lower()
    return any(n in lowered for n in needles)


def _contains_all(text: str, needles: list[str]) -> bool:
    lowered = text.lower()
    return all(n in lowered for n in needles)


def _sentence_count(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    parts = re.split(r"(?<=[.!?])\s+", stripped)
    return len([p for p in parts if p.strip()])


def _bullet_lines(text: str) -> list[str]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return [re.sub(r"^[-*•]\s*", "", l) for l in lines if re.match(r"^[-*•]", l)]


def check_T01(answer: str) -> tuple[bool, str]:
    colors_ok = _contains_all(answer, ["red", "green", "blue"])
    reasoning_ok = _contains_any(answer, ["additive", "emit", "emits", "emitted", "light"])
    if colors_ok and reasoning_ok:
        return True, "mentions red/green/blue and additive-light reasoning"
    missing = []
    if not colors_ok:
        missing.append("one or more of red/green/blue")
    if not reasoning_ok:
        missing.append("additive-light-mixing reasoning (vs RYB's subtractive pigment mixing)")
    return False, f"missing: {', '.join(missing)}"


def check_T01b(answer: str) -> tuple[bool, str]:
    subset_ok = "subset" in answer.lower()
    nn_ok = _contains_any(answer, ["neural network", "neural net", "multi-layer", "multilayer"])
    feature_ok = "feature" in answer.lower()
    if subset_ok and nn_ok and feature_ok:
        return True, "states the subset relationship, neural networks, and feature engineering/extraction"
    missing = []
    if not subset_ok:
        missing.append("explicit 'subset' relationship (rubric: 'Must make the subset relationship clear')")
    if not nn_ok:
        missing.append("neural network / multi-layer mention")
    if not feature_ok:
        missing.append("manual vs automatic feature engineering/extraction")
    return False, f"missing: {', '.join(missing)}"


def check_T01c(answer: str) -> tuple[bool, str]:
    volatility_ok = "volatil" in answer.lower()
    ram_use_ok = _contains_any(answer, ["temporary", "active program", "running program", "active data"])
    rom_use_ok = _contains_any(answer, ["firmware", "bios", "permanent"])
    if volatility_ok and ram_use_ok and rom_use_ok:
        return True, "distinguishes volatility and both RAM/ROM use cases"
    missing = []
    if not volatility_ok:
        missing.append("volatility distinction")
    if not ram_use_ok:
        missing.append("RAM's temporary/active-program use case")
    if not rom_use_ok:
        missing.append("ROM's firmware/BIOS/permanent use case")
    return False, f"missing: {', '.join(missing)}"


def check_T02(answer: str) -> tuple[bool, str]:
    number_ok = _contains_any(answer, ["1672", "1,672"])
    words = answer.split()
    shows_work = len(words) > 5
    if number_ok and shows_work:
        return True, "correct final number (1,672) with some work shown"
    if number_ok and not shows_work:
        return False, "correct number but no arithmetic shown/implied (rubric: 'Minor arithmetic shown or implied')"
    return False, "final answer is not 1,672"


def check_T02b(answer: str) -> tuple[bool, str]:
    sugar_ok = _contains_any(answer, ["1.875", "1.88", "1.87"])
    cost_ok = _contains_any(answer, ["4.50", "4.5"])
    if sugar_ok and cost_ok:
        return True, "correct sugar amount (~1.875 cups) and cost ($4.50)"
    missing = []
    if not sugar_ok:
        missing.append("sugar amount (~1.875 cups)")
    if not cost_ok:
        missing.append("total cost ($4.50)")
    return False, f"missing/incorrect: {', '.join(missing)}"


def _mixed_review_check(answer: str, neg_signals: list[str], pos_signals: list[str]) -> tuple[bool, str]:
    lowered = answer.lower()
    if lowered.startswith("negative"):
        return False, "labeled Negative -- rubric: 'A Negative classification does not pass'"
    neg_ok = _contains_any(answer, neg_signals)
    pos_ok = _contains_any(answer, pos_signals)
    if neg_ok and pos_ok:
        return True, "label is not Negative and reason acknowledges both sides"
    side = "negative" if not neg_ok else "positive"
    return False, f"reason does not acknowledge the {side} side -- rubric: 'A reason that acknowledges only one side does not pass'"


def check_T03(answer: str) -> tuple[bool, str]:
    # Broad net rather than the exact source words ("late"/"damaged"): a
    # good answer is free to paraphrase the negative side (e.g. "the initial
    # delivery and packaging issues") instead of quoting the review
    # verbatim, and the check must not punish that.
    return _mixed_review_check(
        answer,
        neg_signals=["late", "damag", "issue", "delivery", "packaging", "complaint"],
        pos_signals=["perfect", "resolved", "support", "work"],
    )


def check_T03b(answer: str) -> tuple[bool, str]:
    return _mixed_review_check(
        answer,
        neg_signals=["dent", "missing", "issue", "packaging", "box"],
        pos_signals=["flawless", "5 minutes", "five minutes", "setup"],
    )


def check_T04(answer: str) -> tuple[bool, str]:
    n = _sentence_count(answer)
    if n != 2:
        return False, f"expected exactly 2 sentences, got {n} -- rubric: 'more or fewer than two sentences... does not pass'"
    opportunity_ok = _contains_any(answer, ["diagnos", "treatment", "monitor", "analy", "predict"])
    challenge_ok = _contains_any(answer, ["interpret", "bias", "privacy", "liab", "regulat"])
    if opportunity_ok and challenge_ok:
        return True, "exactly 2 sentences covering both the opportunity and the challenges"
    missing = []
    if not opportunity_ok:
        missing.append("the opportunity side (diagnosis/treatment/monitoring/analysis)")
    if not challenge_ok:
        missing.append("the challenge side (interpretability/bias/privacy/liability/regulation)")
    return False, f"exactly 2 sentences, but missing: {', '.join(missing)}"


def check_T04b(answer: str) -> tuple[bool, str]:
    bullets = _bullet_lines(answer)
    if len(bullets) != 3:
        return False, f"expected exactly 3 bullets, got {len(bullets)}"
    over_limit = [b for b in bullets if len(b.split()) > 15]
    if over_limit:
        return False, f"{len(over_limit)} bullet(s) exceed 15 words"
    combined = " ".join(bullets)
    benefit_ok = _contains_any(combined, ["flexib", "work-life", "commute"])
    challenge_ok = _contains_any(combined, ["collaborat", "culture", "boundary", "boundaries"])
    response_ok = _contains_any(combined, ["tool", "office", "organisation", "organization"])
    if benefit_ok and challenge_ok and response_ok:
        return True, "exactly 3 bullets, all <=15 words, covering benefits/challenges/response"
    missing = []
    if not benefit_ok:
        missing.append("benefit theme (flexibility/work-life balance)")
    if not challenge_ok:
        missing.append("challenge theme (collaboration/culture/boundaries)")
    if not response_ok:
        missing.append("organisational response theme (tools/office)")
    return False, f"format OK, but missing theme(s): {', '.join(missing)}"


def check_T05(answer: str) -> tuple[bool, str]:
    try:
        parsed = json.loads(answer)
    except json.JSONDecodeError:
        candidate = re.search(r"\[.*\]", answer, re.DOTALL)
        if not candidate:
            return False, "answer is not valid JSON and no JSON array found"
        try:
            parsed = json.loads(candidate.group(0))
        except json.JSONDecodeError:
            return False, "answer is not valid JSON"

    if not isinstance(parsed, list):
        return False, "parsed JSON is not a list"

    expected = [
        ("sundar pichai", "PERSON"),
        ("march 15 2023", "DATE"),
        ("google", "ORGANIZATION"),
        ("zurich", "LOCATION"),
        ("eth zurich", "ORGANIZATION"),
    ]
    found_texts = {str(item.get("text", "")).lower(): str(item.get("type", "")).upper() for item in parsed if isinstance(item, dict)}

    missing = []
    mislabeled = []
    for exp_text, exp_type in expected:
        match = next((t for t in found_texts if exp_text in t or t in exp_text), None)
        if match is None:
            missing.append(exp_text)
            continue
        actual_type = found_texts[match]
        # Accept ORG as a reasonable abbreviation of ORGANIZATION even though
        # the sample's Expected column spells it out -- but flag it as a
        # format note, not a hard fail, since an LLM judge is likely lenient
        # here. Anything else is a real mislabel.
        if actual_type not in (exp_type, "ORG" if exp_type == "ORGANIZATION" else exp_type):
            mislabeled.append(f"{exp_text} labeled {actual_type} (expected {exp_type})")

    uses_org_abbreviation = any(t == "ORG" for t in found_texts.values())

    if not missing and len(mislabeled) <= 1:
        note = " (uses 'ORG' instead of 'ORGANIZATION' -- matches context.md's schema but not judge_guide.md's spelled-out example; low risk but worth matching exactly)" if uses_org_abbreviation else ""
        return True, f"all 5 entities found, at most 1 mislabeled{note}"
    problems = []
    if missing:
        problems.append(f"missing entities: {', '.join(missing)}")
    if mislabeled:
        problems.append(f"mislabeled: {', '.join(mislabeled)}")
    return False, "; ".join(problems)


CHECKS = {
    "T01": check_T01,
    "T01b": check_T01b,
    "T01c": check_T01c,
    "T02": check_T02,
    "T02b": check_T02b,
    "T03": check_T03,
    "T03b": check_T03b,
    "T04": check_T04,
    "T04b": check_T04b,
    "T05": check_T05,
}


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python3 tests/check_judge_samples.py path/to/results.json", file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1]) as f:
        results = json.load(f)

    by_id = {r["task_id"]: r.get("answer", "") for r in results}

    passed = 0
    total = 0
    print("=" * 72)
    print("JUDGE SAMPLE SELF-CHECK (mechanical approximation, not the real judge)")
    print("=" * 72)
    for task_id, check_fn in CHECKS.items():
        if task_id not in by_id:
            print(f"[SKIP] {task_id}: no answer found in results file")
            continue
        total += 1
        ok, reason = check_fn(by_id[task_id])
        passed += int(ok)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {task_id}: {reason}")

    print("-" * 72)
    print(f"{passed}/{total} passed")
    print("=" * 72)


if __name__ == "__main__":
    main()
