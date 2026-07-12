"""
Self-check for the NER type-vocabulary fix: the system prompt no longer
hardcodes a fixed type list (PERSON/ORGANIZATION/LOCATION/DATE) -- it now
defers to whatever types the *task prompt* specifies, falling back to
standard types only when the prompt doesn't say. This checks that the
model actually follows prompt-specified types instead of always reverting
to the old hardcoded defaults, using input/tasks_ner_type_variety.json.

Not a pytest test (deliberately not named test_*.py), matching the
convention in check_model_availability.py / check_judge_samples.py.

Usage:
    python3 tests/check_ner_type_variety.py path/to/results.json

Generate results first, e.g.:
    set -a && source .env && set +a
    python3 tests/run_fireworks_pipeline.py \\
        --input input/tasks_ner_type_variety.json --output output/ner_variety_results.json
    python3 tests/check_ner_type_variety.py output/ner_variety_results.json
"""
import json
import sys


def _parse(answer: str):
    try:
        parsed = json.loads(answer)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, list) else None


def check_standard_types(answer: str) -> tuple[bool, str]:
    parsed = _parse(answer)
    if parsed is None:
        return False, "not a valid JSON array"
    types_found = {str(item.get("type", "")).upper() for item in parsed if isinstance(item, dict)}
    expected = {"PERSON", "ORGANIZATION", "LOCATION", "DATE"}
    if types_found & expected:
        return True, f"used expected standard types: {sorted(types_found & expected)}"
    return False, f"none of the standard types found; got {sorted(types_found)}"


def check_custom_types(answer: str) -> tuple[bool, str]:
    parsed = _parse(answer)
    if parsed is None:
        return False, "not a valid JSON array"
    types_found = {str(item.get("type", "")).upper() for item in parsed if isinstance(item, dict)}
    custom = {"PRODUCT", "COMPANY", "PRICE"}
    stale_defaults = {"PERSON", "ORGANIZATION", "LOCATION", "DATE"}
    used_custom = types_found & custom
    used_stale = types_found & stale_defaults
    if used_custom and not used_stale:
        return True, f"followed the prompt's custom types: {sorted(used_custom)}"
    if used_stale:
        return False, f"fell back to hardcoded default types instead of the prompt's custom ones: {sorted(used_stale)}"
    return False, f"did not use any of the requested custom types; got {sorted(types_found)}"


def check_subset_only(answer: str) -> tuple[bool, str]:
    parsed = _parse(answer)
    if parsed is None:
        return False, "not a valid JSON array"
    types_found = {str(item.get("type", "")).upper() for item in parsed if isinstance(item, dict)}
    allowed = {"PERSON", "LOCATION"}
    extra = types_found - allowed
    if not types_found:
        return False, "no entities found"
    if extra:
        return False, f"included types outside the requested subset: {sorted(extra)}"
    return True, f"only used the requested subset: {sorted(types_found)}"


def check_no_types_specified(answer: str) -> tuple[bool, str]:
    parsed = _parse(answer)
    if parsed is None:
        return False, "not a valid JSON array"
    if not parsed:
        return False, "no entities found"
    has_text_and_type = all(
        isinstance(item, dict) and item.get("text") and item.get("type") for item in parsed
    )
    if has_text_and_type:
        return True, f"produced {len(parsed)} entities with sensible default types"
    return False, "entities missing text/type fields"


CHECKS = {
    "NER01_standard_types": check_standard_types,
    "NER02_custom_types": check_custom_types,
    "NER03_subset_only": check_subset_only,
    "NER04_no_types_specified": check_no_types_specified,
}


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python3 tests/check_ner_type_variety.py path/to/results.json", file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1]) as f:
        results = json.load(f)
    by_id = {r["task_id"]: r.get("answer", "") for r in results}

    passed = 0
    total = 0
    print("=" * 72)
    print("NER TYPE-VARIETY SELF-CHECK")
    print("=" * 72)
    for task_id, check_fn in CHECKS.items():
        if task_id not in by_id:
            print(f"[SKIP] {task_id}: no answer found in results file")
            continue
        total += 1
        ok, reason = check_fn(by_id[task_id])
        passed += int(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {task_id}: {reason}")
    print("-" * 72)
    print(f"{passed}/{total} passed")
    print("=" * 72)


if __name__ == "__main__":
    main()
