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
import sys

from config import ROLE_CANDIDATE_TIERS, CATEGORY_ROLE
from categories.task_categories import CODE_EXEC_CATEGORIES, LOGIC_NL_CATEGORIES


def get_allowed_models():
    raw = os.environ.get("ALLOWED_MODELS", "")
    models = [m.strip() for m in raw.split(",") if m.strip()]
    if not models:
        raise RuntimeError("ALLOWED_MODELS env var is empty or unset -- cannot route any calls.")
    return models


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

    # Explicit, auditable confirmation of the skip-unhealthy-tiers behavior --
    # makes it obvious in logs whether a role's top-choice model was actually
    # used, or silently skipped because the health probe marked it down.
    for role in roles_needed:
        tiers = per_role_tier_models[role]
        top_choice = tiers[0] if tiers else None
        resolved = _RESOLVED_ROLE_MODEL[role]
        if top_choice and top_choice != resolved:
            print(
                f"[routing] {role}: top-choice model '{top_choice}' is DOWN -- "
                f"routing directly to '{resolved}', skipping it entirely for all tasks this run.",
                file=sys.stderr,
            )
    # Build the ordered summarization cascade from *all* matching models
    # (regardless of health) so the fallback path can walk up the tier list.
    _build_summarization_cascade(allowed)

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


# Ordered list of resolved models for the summarization cascade.
# Each call to route_summarization_next(current_model) returns the next
# healthier/better model in the tier sequence, or None if already at the top.
_SUMMARIZATION_CASCADE: list = []


def _build_summarization_cascade(allowed):
    """Populate the in-order cascade list from the summarization role tiers."""
    global _SUMMARIZATION_CASCADE
    cascade = []
    for tier in ROLE_CANDIDATE_TIERS.get("summarization", []):
        m = _match_in_allowed(tier, allowed)
        if m and m not in cascade:
            cascade.append(m)
    _SUMMARIZATION_CASCADE = cascade


def route_summarization_next(current_model: str) -> str | None:
    """Return the next model in the summarization cascade after *current_model*.

    Returns None if current_model is already the last (best) option.
    Falls back to the full allowed list if the cascade is empty.
    """
    cascade = _SUMMARIZATION_CASCADE
    if not cascade:
        # resolve_roles wasn't called — build on the fly.
        try:
            _build_summarization_cascade(get_allowed_models())
            cascade = _SUMMARIZATION_CASCADE
        except RuntimeError:
            return None

    try:
        idx = cascade.index(current_model)
    except ValueError:
        # current_model isn't in the cascade — start from the beginning.
        return cascade[0] if cascade else None

    if idx + 1 < len(cascade):
        return cascade[idx + 1]
    return None  # already at the best model
