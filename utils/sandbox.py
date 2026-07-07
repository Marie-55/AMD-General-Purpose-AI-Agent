from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


FORBIDDEN_NAMES = {
    "breakpoint",
    "compile",
    "delattr",
    "dir",
    "eval",
    "exec",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
    "__import__",
}

SAFE_IMPORTS = {
    "ast",
    "bisect",
    "collections",
    "datetime",
    "decimal",
    "fractions",
    "functools",
    "heapq",
    "itertools",
    "json",
    "math",
    "operator",
    "random",
    "re",
    "statistics",
    "string",
}


@dataclass(frozen=True)
class SandboxResult:
    stdout: str
    stderr: str
    returncode: int


class SandboxViolation(RuntimeError):
    pass


class ExecutionSandbox:
    """Run model-generated Python in a separate process with a tiny builtin surface."""

    def __init__(self, timeout_s: int = 4):
        self.timeout_s = timeout_s

    def _validate_source(self, source: str) -> None:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".", 1)[0]
                    if root not in SAFE_IMPORTS:
                        raise SandboxViolation(f"Import not allowed: {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module is None:
                    continue
                root = node.module.split(".", 1)[0]
                if root not in SAFE_IMPORTS:
                    raise SandboxViolation(f"Import not allowed: {node.module}")
            elif isinstance(node, ast.Name):
                if node.id in FORBIDDEN_NAMES:
                    raise SandboxViolation(f"Forbidden name used in sandboxed code: {node.id}")
                if node.id.startswith("__"):
                    raise SandboxViolation(f"Dunder names are not allowed in sandboxed code: {node.id}")
            elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
                raise SandboxViolation(f"Dunder attributes are not allowed in sandboxed code: {node.attr}")

    def run(self, source: str) -> SandboxResult:
        self._validate_source(source)
        wrapper = f"""
import builtins
import collections
import datetime
import decimal
import fractions
import functools
import heapq
import itertools
import json
import math
import operator
import random
import re
import statistics
import string

SAFE_MODULES = {{
    "ast": __import__("ast"),
    "bisect": __import__("bisect"),
    "collections": collections,
    "datetime": datetime,
    "decimal": decimal,
    "fractions": fractions,
    "functools": functools,
    "heapq": heapq,
    "itertools": itertools,
    "json": json,
    "math": math,
    "operator": operator,
    "random": random,
    "re": re,
    "statistics": statistics,
    "string": string,
}}

SAFE_BUILTINS = {{
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "next": next,
    "range": range,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}}


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".", 1)[0]
    if root not in SAFE_MODULES:
        raise ImportError(f"Import not allowed: {{name}}")
    return SAFE_MODULES[root]


SAFE_BUILTINS["__import__"] = _safe_import
SAFE_BUILTINS["print"] = builtins.print

sandbox_globals = {{"__builtins__": SAFE_BUILTINS}}
source = {json.dumps(source)}
exec(compile(source, "<sandbox>", "exec"), sandbox_globals, sandbox_globals)

for name in ("answer", "result", "output"):
    if name in sandbox_globals and sandbox_globals[name] is not None:
        value = sandbox_globals[name]
        if isinstance(value, str):
            print(value)
        else:
            print(json.dumps(value, ensure_ascii=False))
        break
"""
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as handle:
            script_path = Path(handle.name)
            handle.write(wrapper)

        try:
            completed = subprocess.run(
                [sys.executable, "-I", str(script_path)],
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                env={"PYTHONNOUSERSITE": "1"},
            )
        finally:
            script_path.unlink(missing_ok=True)

        return SandboxResult(
            stdout=completed.stdout.strip(),
            stderr=completed.stderr.strip(),
            returncode=completed.returncode,
        )
