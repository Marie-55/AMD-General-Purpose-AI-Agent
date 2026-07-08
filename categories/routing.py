"""
Static model routing with a one-time cascade health check.

Why: some allowed models may be registered but not actually servable on a
given account/environment (e.g. dedicated-deployment-only models returning
404 on chat completions even though they show as READY in the catalog).
Rather than hardcoding "skip Gemma" -- which could be wrong on eval day if
the harness's own credentials serve them fine -- we probe every candidate
model ONCE at pipeline startup with a minimal 1-token call, cache which
ones actually respond, and route the rest of the run off that cache.

Cost/latency: probes run concurrently (one round-trip, not N), and each
probe uses max_tokens=1, so this adds roughly one request's worth of time
to the whole run, not one per task.
"""
import os
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_allowed_models():
    raw = os.environ.get("ALLOWED_MODELS", "")
    models = [m.strip() for m in raw.split(",") if m.strip()]
    if not models:
        raise RuntimeError("ALLOWED_MODELS env var is empty or unset -- cannot route any calls.")
    return models


# Each role lists candidate keyword-tiers in priority order. The first tier
# whose matching model passes the health probe wins. Tiers after the first
# are deliberate fallbacks, not "any allowed model" -- kept short and
# specific so we never fall back into an expensive/wrong-fit model.
ROLE_CANDIDATE_TIERS = {
    # cheap categories (facts, sentiment, NER): cheapest model first,
    # fall back to the general-purpose model (not the code specialist,
    # which tends to spend extra reasoning tokens on non-code prompts).
    "cheap_general": [
        ["gemma-4-26b-a4b-it", "a4b"],
        ["minimax-m3", "minimax"],
    ],
    "cheap_alt": [
        ["gemma-4-31b-it-nvfp4", "nvfp4"],
        ["minimax-m3", "minimax"],
    ],
    "quality_general": [
        ["gemma-4-31b-it"],
        ["minimax-m3", "minimax"],
    ],
    # code categories: no cheaper substitute makes sense here, so a single
    # tier -- if the code specialist is down we want the direct-answer
    # fallback in main.py to kick in, not a silent swap to a weaker fit.
    "code_specialist": [
        ["kimi", "code"],
    ],
    "reasoning_specialist": [
        ["minimax"],
    ],
}

CATEGORY_ROLE = {
    "factual_knowledge": "cheap_general",
    "sentiment": "cheap_general",
    "summarization": "cheap_alt",
    "ner": "cheap_general",
    "code_debugging": "code_specialist",
    "code_generation": "code_specialist",
    "math_reasoning": "code_specialist",
    "logic_puzzle": "reasoning_specialist",
}

CODE_EXEC_CATEGORIES = {"math_reasoning", "logic_puzzle"}

# Populated once by resolve_roles(); route() reads from this cache.
_RESOLVED_ROLE_MODEL = {}
_HEALTH_CACHE = {}


def _match_in_allowed(keywords, allowed):
    for kw in keywords:
        for m in allowed:
            if kw in m:
                return m
    return None


def _probe(client, model, timeout=6.0):
    if model in _HEALTH_CACHE:
        return _HEALTH_CACHE[model]
    try:
        _, lat = client.call(model, [{"role": "user", "content": "hi"}], max_tokens=1, timeout=timeout)
        ok = not lat.get("error")
    except Exception:
        ok = False
    _HEALTH_CACHE[model] = ok
    return ok


def resolve_roles(client, allowed=None, timeout=6.0):
    """
    Run once at startup. Probes every distinct candidate model referenced
    by CATEGORY_ROLE (in tier order) concurrently, then picks -- per role --
    the first tier whose model passed. Falls back to allowed[0] for a role
    if nothing in any tier is reachable, so we never crash the run.
    """
    if allowed is None:
        allowed = get_allowed_models()

    roles_needed = set(CATEGORY_ROLE.values())

    # Collect every distinct model we might need to probe, across all roles/tiers.
    candidate_models = set()
    per_role_tier_models = {}
    for role in roles_needed:
        tier_models = []
        for tier in ROLE_CANDIDATE_TIERS.get(role, []):
            m = _match_in_allowed(tier, allowed)
            tier_models.append(m)
            if m:
                candidate_models.add(m)
        per_role_tier_models[role] = tier_models

    # Probe all distinct models concurrently -- one round-trip, not N.
    with ThreadPoolExecutor(max_workers=max(1, len(candidate_models))) as pool:
        futures = {pool.submit(_probe, client, m, timeout): m for m in candidate_models}
        for fut in as_completed(futures):
            fut.result()  # populates _HEALTH_CACHE; errors already swallowed in _probe

    for role in roles_needed:
        chosen = None
        for m in per_role_tier_models[role]:
            if m and _HEALTH_CACHE.get(m):
                chosen = m
                break
        _RESOLVED_ROLE_MODEL[role] = chosen or allowed[0]

    return dict(_RESOLVED_ROLE_MODEL), dict(_HEALTH_CACHE)


def route(category: str) -> str:
    role = CATEGORY_ROLE.get(category, "cheap_general")
    if role in _RESOLVED_ROLE_MODEL:
        return _RESOLVED_ROLE_MODEL[role]
    # Safety net if resolve_roles() wasn't called (e.g. unit tests calling
    # route() directly): fail safe onto the first allowed model.
    allowed = get_allowed_models()
    for tier in ROLE_CANDIDATE_TIERS.get(role, []):
        m = _match_in_allowed(tier, allowed)
        if m:
            return m
    return allowed[0]