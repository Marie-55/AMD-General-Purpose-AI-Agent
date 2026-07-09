from __future__ import annotations

from utils.heuristics import is_retryable_model_error, rank_models
from utils.local_solvers import local_answer_for_category
from utils.models import RuntimeConfig, Task, TaskCategory, UsageTotals
from utils.local_model import LocalModelClient
from utils.parsing import (
    collapse_json_answer,
    compact_text,
    finalize_code_answer,
    finalize_plain_answer,
    parse_json_loose,
    sanitize_final_answer,
    strip_code_fences,
    summarization_budget,
)
from utils.prompting import compact_task_prompt, compact_user_prompt


class CategorySolvers:
    """Category-specific execution paths for the track 1 prompt families."""

    def __init__(
        self,
        client,
        config: RuntimeConfig,
        sandbox,
        local_client: LocalModelClient | None = None,
        usage: UsageTotals | None = None,
    ):
        self.client = client
        self.config = config
        self.sandbox = sandbox
        self.local_client = local_client
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
        user_prompt = compact_task_prompt(user_prompt, category=role, budget_chars=1200)
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

    def _call_local_text_model(
        self,
        role: str,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> str:
        if self.local_client is None or not self.local_client.available:
            raise RuntimeError("Local model is unavailable.")
        user_prompt = compact_task_prompt(user_prompt, category=role, budget_chars=1200)
        text, _, _ = self.local_client.chat_completion(
            "local-model",
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return text

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
            repair_prompt = compact_user_prompt(
                f"Rewrite into strict JSON only with keys: {', '.join(expected_keys)}.",
                f"Original output:\n{text}",
                category=role,
                budget_chars=500,
            )
            repaired = self._call_text_model(role, system_prompt, repair_prompt, max_tokens=max_tokens)
            data = parse_json_loose(repaired)
            if not isinstance(data, dict):
                raise ValueError("JSON repair did not produce an object.")
            return data

    def solve(
        self,
        task: Task,
        category: str,
        *,
        allow_remote: bool = True,
        prefer_local_model: bool = False,
    ) -> str:
        if category == TaskCategory.SUMMARY.value:
            return self.solve_summary(task, allow_remote=allow_remote, prefer_local_model=prefer_local_model)
        if category == TaskCategory.SENTIMENT.value:
            return self.solve_sentiment(task, allow_remote=allow_remote)
        if category == TaskCategory.NER.value:
            return self.solve_ner(task, allow_remote=allow_remote)
        if category == TaskCategory.DEBUGGING.value:
            return self.solve_code_debugging(task, allow_remote=allow_remote)
        if category == TaskCategory.CODE.value:
            return self.solve_code_generation(task, allow_remote=allow_remote)
        if category == TaskCategory.LOGIC.value:
            return self.solve_program_aided(task, label="logic", allow_remote=allow_remote)
        if category == TaskCategory.MATH.value:
            return self.solve_program_aided(task, label="math", allow_remote=allow_remote, prefer_local_model=prefer_local_model)
        if category == TaskCategory.FACTUAL.value:
            return self.solve_factual(task, allow_remote=allow_remote, prefer_local_model=prefer_local_model)
        return self.solve_factual(task, allow_remote=allow_remote, prefer_local_model=prefer_local_model)

    def solve_factual(
        self,
        task: Task,
        *,
        allow_remote: bool = True,
        prefer_local_model: bool = False,
    ) -> str:
        if prefer_local_model and self.local_client is not None and self.local_client.available:
            system = "Answer fact questions concisely. Return only the final answer."
            try:
                text = self._call_local_text_model("factual", system, task.prompt, max_tokens=96)
                compacted = finalize_plain_answer(text)
                if compacted:
                    return compacted
            except Exception:
                pass
        local = local_answer_for_category(TaskCategory.FACTUAL.value, task.prompt)
        if local:
            return finalize_plain_answer(local)
        if not allow_remote:
            return finalize_plain_answer(local or "")
        system = "Answer fact questions concisely. Return only the final answer."
        text = self._call_text_model("factual", system, task.prompt, max_tokens=96)
        return finalize_plain_answer(text)

    def solve_summary(
        self,
        task: Task,
        *,
        allow_remote: bool = True,
        prefer_local_model: bool = False,
    ) -> str:
        if prefer_local_model and self.local_client is not None and self.local_client.available:
            system = "Summarize faithfully. Return only the summary."
            try:
                text = self._call_local_text_model("summarization", system, task.prompt, max_tokens=min(self.config.local_model_max_tokens, 160))
                compacted = finalize_plain_answer(text)
                if compacted:
                    return compacted
            except Exception:
                pass
        local = local_answer_for_category(TaskCategory.SUMMARY.value, task.prompt)
        if local:
            return finalize_plain_answer(local)
        if not allow_remote:
            return finalize_plain_answer(local or "")
        max_tokens = summarization_budget(task.prompt)
        system = "Summarize faithfully. Return only the summary."
        text = self._call_text_model("summarization", system, task.prompt, max_tokens=min(max_tokens, 128))
        return finalize_plain_answer(text)

    def solve_sentiment(self, task: Task, *, allow_remote: bool = True) -> str:
        local = local_answer_for_category(TaskCategory.SENTIMENT.value, task.prompt)
        if local:
            return compact_text(local)
        if not allow_remote:
            return compact_text(local or "")
        system = "Classify sentiment and return JSON only."
        user = (
            f"{task.prompt}\n\n"
            'Return JSON exactly in this shape: {"label":"positive|negative|neutral","reason":"short justification"}'
        )
        data = self._call_json_model("sentiment", system, user, max_tokens=64, expected_keys=["label", "reason"])
        label = str(data.get("label", "")).strip().lower()
        reason = str(data.get("reason", "")).strip()
        if label not in {"positive", "negative", "neutral"}:
            label = "neutral"
        return collapse_json_answer({"label": label, "reason": reason})

    def solve_ner(self, task: Task, *, allow_remote: bool = True) -> str:
        local = local_answer_for_category(TaskCategory.NER.value, task.prompt)
        if local:
            return compact_text(local)
        if not allow_remote:
            return compact_text(local or "")
        system = "Extract named entities and return JSON only."
        user = (
            f"{task.prompt}\n\n"
            'Return JSON exactly in this shape: {"entities":[{"text":"...","type":"person|org|location|date|other"}]}'
        )
        data = self._call_json_model("ner", system, user, max_tokens=96, expected_keys=["entities"])
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

    def solve_code_debugging(self, task: Task, *, allow_remote: bool = True) -> str:
        local = local_answer_for_category(TaskCategory.DEBUGGING.value, task.prompt)
        if local:
            return finalize_code_answer(local)
        if not allow_remote:
            return ""
        system = "Fix the code. Return code only."
        text = self._call_text_model("code", system, task.prompt, max_tokens=384)
        return finalize_code_answer(text)

    def solve_code_generation(self, task: Task, *, allow_remote: bool = True) -> str:
        local = local_answer_for_category(TaskCategory.CODE.value, task.prompt)
        if local:
            return finalize_code_answer(local)
        if not allow_remote:
            return ""
        system = "Write correct code. Return code only."
        text = self._call_text_model("code", system, task.prompt, max_tokens=384)
        return finalize_code_answer(text)

    def solve_program_aided(
        self,
        task: Task,
        *,
        label: str,
        allow_remote: bool = True,
        prefer_local_model: bool = False,
    ) -> str:
        if prefer_local_model and self.local_client is not None and self.local_client.available:
            system = "Answer directly. Return only the final answer."
            try:
                text = self._call_local_text_model("math", system, task.prompt, max_tokens=self.config.local_model_max_tokens)
                compacted = finalize_plain_answer(text)
                if compacted:
                    return compacted
            except Exception:
                pass

        local = local_answer_for_category(TaskCategory.MATH.value if label == "math" else TaskCategory.LOGIC.value, task.prompt)
        if local:
            return finalize_plain_answer(local)

        if not allow_remote:
            return finalize_plain_answer(local or "")

        primary_system = "Write a compact Python 3 program. Return code only."
        primary_user = compact_user_prompt(
            "Return only executable Python code. Keep it short and deterministic.",
            task.prompt,
            category=label,
            budget_chars=1200,
        )
        code_text = self._call_text_model("code", primary_system, primary_user, max_tokens=320)

        source = strip_code_fences(code_text)
        source = finalize_code_answer(source)
        sandbox_result = self.sandbox.run(source)
        if sandbox_result.returncode == 0 and sandbox_result.stdout:
            return sanitize_final_answer(sandbox_result.stdout)

        repair_system = "Repair the Python program. Return code only."
        repair_user = compact_user_prompt(
            f"Problem label: {label}. Fix the program using the sandbox output.",
            (
                f"Original prompt:\n{task.prompt}\n\n"
                f"Previous code:\n{source}\n\n"
                f"Sandbox stdout:\n{sandbox_result.stdout}\n\n"
                f"Sandbox stderr:\n{sandbox_result.stderr}"
            ),
            category=label,
            budget_chars=1400,
        )
        repair_text = self._call_text_model("code", repair_system, repair_user, max_tokens=256)
        repaired_source = finalize_code_answer(repair_text)
        repaired_result = self.sandbox.run(repaired_source)
        if repaired_result.returncode == 0 and repaired_result.stdout:
            return sanitize_final_answer(repaired_result.stdout)

        direct_system = "Answer directly. Return only the final answer."
        direct_text = self._call_text_model("math", direct_system, task.prompt, max_tokens=96)
        return finalize_plain_answer(direct_text)

    def solve_fallback(self, task: Task, error: Exception | None = None, *, allow_remote: bool = True) -> str:
        strongest_role = "code" if any(
            keyword in task.prompt.lower() for keyword in ("code", "function", "debug", "bug")
        ) else "math"
        system = "Return only the final answer in the exact requested format."
        if error is not None:
            user = f"{task.prompt}\n\nA previous attempt failed with: {error!s}"
        else:
            user = task.prompt
        if not allow_remote:
            return finalize_plain_answer(user)
        text = self._call_text_model(strongest_role, system, user, max_tokens=96)
        return finalize_plain_answer(text)
