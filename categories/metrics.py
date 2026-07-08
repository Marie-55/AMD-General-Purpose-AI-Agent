"""
Monitoring / metrics collection.

Tracks, per task: category, routing path, model used, exact request-sent /
first-token / request-end timestamps, time-to-first-token (network+queue
latency proxy), total latency, and prompt/completion/total tokens.

Also maintains a running ESTIMATED dev-credit spend against the $50 personal
benchmarking budget. This estimate is for your OWN dev-key testing only --
it has nothing to do with the harness's token-based ranking (the harness
ranks by total tokens recorded by its proxy, not by dollars). Fill in
PRICE_PER_1M_TOKENS with real Fireworks prices once you have them; until then
treat the cost figures as illustrative, not authoritative.
"""
import time

# $ per 1,000,000 tokens, split input/output. Placeholder values -- update once
# you know Fireworks' actual per-model pricing for the allowed models.
PRICE_PER_1M_TOKENS = {
    "default": {"input": 0.20, "output": 0.80},
}


class MetricsCollector:
    def __init__(self, credit_budget_usd: float = 50.0):
        self.records = []
        self.credit_budget_usd = credit_budget_usd
        self.credit_spent_usd = 0.0
        self.start_time = time.time()

    def log(self, task_id, category, path, model, latency_metrics):
        rec = {
            "task_id": task_id,
            "category": category,
            "path": path,  # "direct_llm" | "code_exec" | "code_exec_fallback"
            "model": model,
            **latency_metrics,
        }
        cost = self._estimate_cost(model, latency_metrics)
        rec["estimated_cost_usd"] = cost
        self.credit_spent_usd += cost
        self.records.append(rec)
        return rec

    def _estimate_cost(self, model, m):
        price = PRICE_PER_1M_TOKENS.get(model, PRICE_PER_1M_TOKENS["default"])
        pt = m.get("prompt_tokens") or 0
        ct = m.get("completion_tokens") or 0
        return round(pt / 1e6 * price["input"] + ct / 1e6 * price["output"], 6)

    def summary(self):
        total_tokens = sum(r.get("total_tokens") or 0 for r in self.records)
        total_prompt = sum(r.get("prompt_tokens") or 0 for r in self.records)
        total_completion = sum(r.get("completion_tokens") or 0 for r in self.records)
        elapsed = time.time() - self.start_time
        errors = [r for r in self.records if r.get("error")]

        by_cat = {}
        for r in self.records:
            d = by_cat.setdefault(r["category"], {"count": 0, "tokens": 0, "avg_latency_ms": 0.0})
            d["count"] += 1
            d["tokens"] += r.get("total_tokens") or 0
            d["avg_latency_ms"] += r.get("total_latency_ms") or 0
        for d in by_cat.values():
            if d["count"]:
                d["avg_latency_ms"] = round(d["avg_latency_ms"] / d["count"], 1)

        return {
            "elapsed_s": round(elapsed, 2),
            "num_tasks": len(self.records),
            "num_errors": len(errors),
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_tokens": total_tokens,
            "estimated_credit_spent_usd": round(self.credit_spent_usd, 4),
            "estimated_credit_remaining_usd": round(self.credit_budget_usd - self.credit_spent_usd, 4),
            "by_category": by_cat,
        }

    def error_records(self):
        return [r for r in self.records if r.get("error")]

    def print_report(self):
        s = self.summary()
        print("=" * 64)
        print("RUN SUMMARY")
        print("=" * 64)
        for k, v in s.items():
            if k != "by_category":
                print(f"{k:35s}: {v}")
        print("-- by category --")
        for cat, d in s["by_category"].items():
            print(f"  {cat:20s} count={d['count']:3d}  tokens={d['tokens']:6d}  avg_latency_ms={d['avg_latency_ms']}")
        errors = self.error_records()
        if errors:
            print("-- errors (first 10) --")
            for r in errors[:10]:
                msg = str(r.get("error") or "")
                msg = msg.replace("\n", " ")[:500]
                print(f"  task_id={r.get('task_id')} model={r.get('model')} path={r.get('path')} error={msg}")
        print("=" * 64)

    def print_per_call_table(self, limit=None):
        rows = self.records if limit is None else self.records[:limit]
        header = f"{'task_id':8s} {'category':18s} {'path':18s} {'model':22s} {'ttft_ms':8s} {'total_ms':9s} {'tokens':7s}"
        print(header)
        print("-" * len(header))
        for r in rows:
            print(
                f"{r['task_id']:8s} {r['category']:18s} {r['path']:18s} "
                f"{r['model']:22s} {str(r.get('ttft_ms')):8s} {str(r.get('total_latency_ms')):9s} "
                f"{str(r.get('total_tokens')):7s}"
            )