from __future__ import annotations

from utils.heuristics import is_retryable_model_error, rank_models
from utils.models import RuntimeConfig, Task, TaskCategory, UsageTotals
from utils.parsing import (
    collapse_json_answer,
    compact_text,
    parse_json_loose,
    sanitize_final_answer,
    strip_code_fences,
    summarization_budget,
)


class CategorySolvers:
    """Category-specific execution paths for the track 1 prompt families."""

    def __init__(
        self,
        client,
        config: RuntimeConfig,
        sandbox,
        usage: UsageTotals | None = None,
    ):
        self.client = client
        self.config = config
        self.sandbox = sandbox
        self.usage = usage

    def _record_usage(self, usage: tuple[int, int]) -> None:
        if self.usage is not None:
            self.usage.add(*usage)

    def _call_text_model(
        self,
        role: str,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> str:
        last_error: Exception | None = None
        for model in rank_models(self.config.allowed_models, role):
            try:
                text, usage, _ = self.client.chat_completion(
                    model,
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                self._record_usage(usage)
                return text
            except Exception as exc:
                last_error = exc
                if not is_retryable_model_error(exc):
                    raise
        raise RuntimeError(f"All {role} models failed: {last_error}")

    def _call_json_model(
        self,
        role: str,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int,
        expected_keys: list[str],
    ) -> dict[str, object]:
        text = self._call_text_model(role, system_prompt, user_prompt, max_tokens=max_tokens)
        try:
            data = parse_json_loose(text)
            if not isinstance(data, dict):
                raise ValueError("JSON output was not an object.")
            return data
        except Exception:
            repair_prompt = (
                "Rewrite the following into strict JSON only. "
                f"Ensure the object contains these keys: {', '.join(expected_keys)}.\n\n"
                f"Original output:\n{text}"
            )
            repaired = self._call_text_model(role, system_prompt, repair_prompt, max_tokens=max_tokens)
            data = parse_json_loose(repaired)
            if not isinstance(data, dict):
                raise ValueError("JSON repair did not produce an object.")
            return data

    def solve(self, task: Task, category: str) -> str:
        if category == TaskCategory.SUMMARY.value:
            return self.solve_summary(task)
        if category == TaskCategory.SENTIMENT.value:
            return self.solve_sentiment(task)
        if category == TaskCategory.NER.value:
            return self.solve_ner(task)
        if category == TaskCategory.DEBUGGING.value:
            return self.solve_code_debugging(task)
        if category == TaskCategory.CODE.value:
            return self.solve_code_generation(task)
        if category == TaskCategory.LOGIC.value:
            return self.solve_program_aided(task, label="logic")
        if category == TaskCategory.MATH.value:
            return self.solve_program_aided(task, label="math")
        if category == TaskCategory.FACTUAL.value:
            return self.solve_factual(task)
        return self.solve_factual(task)

    def solve_factual(self, task: Task) -> str:
        system = (
            "You answer factual questions concisely and accurately. "
            "Use plain English, avoid chain-of-thought, and keep the answer compact."
        )
        text = self._call_text_model("factual", system, task.prompt, max_tokens=256)
        return compact_text(text)

    def solve_summary(self, task: Task) -> str:
        max_tokens = summarization_budget(task.prompt)
        system = (
            "You summarize text faithfully. "
            "Preserve the requested style and length constraints exactly. "
            "Return only the final summary, with no preamble."
        )
        text = self._call_text_model("summarization", system, task.prompt, max_tokens=max_tokens)
        return compact_text(text)

    def solve_sentiment(self, task: Task) -> str:
        system = (
            "Classify sentiment. Return strict JSON only with keys label and reason. "
            "Label must be one of positive, negative, or neutral."
        )
        user = (
            f"{task.prompt}\n\n"
            'Return JSON exactly in this shape: {"label":"positive|negative|neutral","reason":"short justification"}'
        )
        data = self._call_json_model("sentiment", system, user, max_tokens=96, expected_keys=["label", "reason"])
        label = str(data.get("label", "")).strip().lower()
        reason = str(data.get("reason", "")).strip()
        if label not in {"positive", "negative", "neutral"}:
            label = "neutral"
        return collapse_json_answer({"label": label, "reason": reason})

    def solve_ner(self, task: Task) -> str:
        system = (
            "Extract named entities. Return strict JSON only with an entities array. "
            "Each entity should include text and type."
        )
        user = (
            f"{task.prompt}\n\n"
            'Return JSON exactly in this shape: {"entities":[{"text":"...","type":"person|org|location|date|other"}]}'
        )
        data = self._call_json_model("ner", system, user, max_tokens=160, expected_keys=["entities"])
        entities = data.get("entities", [])
        if not isinstance(entities, list):
            entities = []
        normalized_entities: list[dict[str, str]] = []
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            text = str(entity.get("text", "")).strip()
            entity_type = str(entity.get("type", "other")).strip().lower() or "other"
            if text:
                normalized_entities.append({"text": text, "type": entity_type})
        return collapse_json_answer({"entities": normalized_entities})

    def solve_code_debugging(self, task: Task) -> str:
        system = (
            "You are an expert code debugger. Return the corrected code only, with no explanation. "
            "Preserve the original language and structure as much as possible."
        )
        text = self._call_text_model("code", system, task.prompt, max_tokens=768)
        return strip_code_fences(text).strip()

    def solve_code_generation(self, task: Task) -> str:
        system = (
            "You write correct code that satisfies the specification. Return code only, no explanation, no markdown."
        )
        text = self._call_text_model("code", system, task.prompt, max_tokens=768)
        return strip_code_fences(text).strip()

    def solve_program_aided(self, task: Task, *, label: str) -> str:
        primary_system = (
            "Write a compact Python 3 program that solves the problem exactly. "
            "Use only the standard library. "
            "Assign the final answer to a variable named answer, or print it once at the end. "
            "Do not explain your reasoning."
        )
        primary_user = (
            f"{task.prompt}\n\n"
            "Return only executable Python code. Keep it short, deterministic, and self-contained."
        )
        code_text = self._call_text_model("code", primary_system, primary_user, max_tokens=512)

        source = strip_code_fences(code_text)
        sandbox_result = self.sandbox.run(source)
        if sandbox_result.returncode == 0 and sandbox_result.stdout:
            return sanitize_final_answer(sandbox_result.stdout)

        repair_system = (
            "You write a compact Python 3 program that must run without errors in a restricted sandbox. "
            "Use only the standard library. Assign the final answer to answer or print it once. "
            "Return code only."
        )
        repair_user = (
            f"Problem label: {label}\n\n"
            f"Original prompt:\n{task.prompt}\n\n"
            f"Previous code:\n{source}\n\n"
            f"Sandbox stdout:\n{sandbox_result.stdout}\n\n"
            f"Sandbox stderr:\n{sandbox_result.stderr}\n\n"
            "Fix the program."
        )
        repair_text = self._call_text_model("code", repair_system, repair_user, max_tokens=512)
        repaired_source = strip_code_fences(repair_text)
        repaired_result = self.sandbox.run(repaired_source)
        if repaired_result.returncode == 0 and repaired_result.stdout:
            return sanitize_final_answer(repaired_result.stdout)

        direct_system = (
            "Answer the problem directly and concisely. "
            "Return only the final answer, no explanation."
        )
        direct_text = self._call_text_model("math", direct_system, task.prompt, max_tokens=160)
        return compact_text(direct_text)

    def solve_fallback(self, task: Task, error: Exception | None = None) -> str:
        strongest_role = "code" if any(
            keyword in task.prompt.lower() for keyword in ("code", "function", "debug", "bug")
        ) else "math"
        system = (
            "You are a reliable assistant. Return only the final answer in the exact format requested by the user."
        )
        if error is not None:
            user = f"{task.prompt}\n\nA previous attempt failed with: {error!s}"
        else:
            user = task.prompt
        text = self._call_text_model(strongest_role, system, user, max_tokens=256)
        return compact_text(text)
