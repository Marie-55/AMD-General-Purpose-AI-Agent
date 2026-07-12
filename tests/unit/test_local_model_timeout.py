"""
Regression test for the local-model concurrency bug found via a real run:
under concurrent load, every local call routed through
categories.handlers.base.try_local_first() was effectively discarded even
when the model was healthy, because the old implementation wrapped each
call in `with ThreadPoolExecutor(...) as ex: ... fut.result(timeout=8)`.
Catching the TimeoutError there does NOT free the calling thread: exiting
that `with` block still calls shutdown(wait=True), which blocks until the
still-running background thread actually finishes -- so a "timed out" call
paid its full real latency anyway and then threw the result away. Only
math_reasoning (which calls the local model directly, unguarded) was
unaffected, which is why it was the only category that reliably benefited
from local inference under concurrent load.

The fix moves the actual bound to where the real blocking happens:
LocalModelSingleton.generate()'s lock acquisition, via
threading.Lock.acquire(timeout=...), which is a real bounded wait.
"""
import threading
import time

import pytest

from categories.local_model import LocalModelSingleton
from categories.handlers.base import run_local_with_timeout


class _FakeLlm:
    """Callable stand-in for llama_cpp.Llama that blocks for a fixed duration."""

    def __init__(self, delay_s: float):
        self.delay_s = delay_s

    def __call__(self, prompt_str, **kwargs):
        time.sleep(self.delay_s)
        return {"choices": [{"text": " ok "}]}


def _make_instance(delay_s: float) -> LocalModelSingleton:
    # No real GGUF file is available in this environment; construct with
    # paths that don't exist (leaves _llm=None, exactly like a real "model
    # failed to load" case) and then attach a fake _llm directly so the
    # real generate()/lock code path is exercised.
    inst = LocalModelSingleton(primary_path="/nonexistent/a.gguf", fallback_path="/nonexistent/b.gguf")
    inst._llm = _FakeLlm(delay_s)
    return inst


def test_generate_returns_promptly_when_lock_is_free():
    inst = _make_instance(delay_s=0.01)
    assert inst.generate("hi", lock_timeout_s=1.0) == "ok"


def test_generate_raises_timeout_error_quickly_when_lock_is_held():
    inst = _make_instance(delay_s=0.5)
    holder_started = threading.Event()

    def hold_lock():
        inst._lock.acquire()
        holder_started.set()
        time.sleep(0.5)
        inst._lock.release()

    holder = threading.Thread(target=hold_lock)
    holder.start()
    assert holder_started.wait(timeout=1.0)

    start = time.time()
    with pytest.raises(TimeoutError):
        inst.generate("hi", lock_timeout_s=0.1)
    elapsed = time.time() - start
    holder.join()  # cleanup only -- must not count against the measured elapsed time

    # This is the actual bug fix under test: a caller that can't get the
    # lock within lock_timeout_s must return in roughly that time, not
    # after waiting for the lock holder's full remaining duration (~0.5s).
    assert elapsed < 0.4


def test_admission_gate_rejects_extra_callers_when_at_capacity():
    """
    Regression test for the admission-control fix: LocalModelSingleton caps
    concurrent attempts at _MAX_CONCURRENT_ATTEMPTS so that additional
    callers fail fast on the admission gate instead of each queueing for the
    lock and burning most of their own lock_timeout_s doing nothing.
    """
    inst = _make_instance(delay_s=0.3)
    inst._ADMISSION_TIMEOUT_S = 0.05  # keep the test fast and deterministic
    capacity = inst._MAX_CONCURRENT_ATTEMPTS

    started = threading.Barrier(capacity)
    results = {}

    def occupy(name):
        started.wait()
        try:
            inst.generate(f"prompt-{name}", lock_timeout_s=2.0)
            results[name] = "ok"
        except TimeoutError as exc:
            results[name] = f"error: {exc}"

    threads = [threading.Thread(target=occupy, args=(i,)) for i in range(capacity)]
    for t in threads:
        t.start()
    time.sleep(0.1)  # let all threads pass the barrier and occupy every admission slot

    start = time.time()
    with pytest.raises(TimeoutError, match="too many callers"):
        inst.generate("prompt-overflow", lock_timeout_s=2.0)
    elapsed = time.time() - start

    for t in threads:
        t.join()

    # Bounded by the (overridden) admission timeout, not lock_timeout_s.
    assert elapsed < 0.3
    assert all(results[i] == "ok" for i in range(capacity))


def test_run_local_with_timeout_does_not_block_on_a_still_running_call():
    """
    Regression guard for the ThreadPoolExecutor.shutdown(wait=True) trap:
    a fn that raises TimeoutError must cause an immediate '' return, not a
    multi-second wait.
    """
    def busy_local_call(*args):
        raise TimeoutError("Local model busy: timed out waiting for a turn.")

    start = time.time()
    result = run_local_with_timeout(busy_local_call, "prompt", object())
    elapsed = time.time() - start

    assert result == ""
    assert elapsed < 0.1
