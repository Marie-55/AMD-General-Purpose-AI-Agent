"""Loads exactly one GGUF model into memory at a time.

The eval environment has 4 GB RAM -- not enough to hold two models
simultaneously. main.py groups tasks by model and processes each group
as one batch, so each model is loaded at most once per run.
"""
import gc

import config


class ModelRuntime:
    def __init__(self):
        self._llm = None
        self._current_key = None

    def load(self, key: str) -> None:
        if self._current_key == key:
            return
        self.unload()
        from llama_cpp import Llama
        path = config.MODEL_PATHS[key]
        print(f"[runtime] loading {key} from {path}", flush=True)
        self._llm = Llama(
            model_path=path,
            n_ctx=config.N_CTX,
            n_threads=config.N_THREADS,
            n_batch=config.N_BATCH,
            verbose=False,
        )
        self._current_key = key
        print(f"[runtime] {key} ready", flush=True)

    def unload(self) -> None:
        if self._llm is not None:
            del self._llm
            self._llm = None
            self._current_key = None
            gc.collect()

    def generate(self, messages, max_tokens: int = 300, temperature: float = 0.2, grammar=None) -> str:
        if self._llm is None:
            raise RuntimeError("No model loaded -- call load() first")
        kwargs = dict(messages=messages, max_tokens=max_tokens, temperature=temperature)
        if grammar is not None:
            kwargs["grammar"] = grammar
        try:
            out = self._llm.create_chat_completion(**kwargs)
            return out["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            print(f"[model_runtime] generation failed: {exc}", flush=True)
            return ""
