import json

from categories.code_exec import extract_code, run_code_safely


def test_extract_code_from_fenced_block():
    llm_output = "Here you go:\n```python\nx = 1\nprint(x)\n```\nDone."
    assert extract_code(llm_output) == "x = 1\nprint(x)"


def test_extract_code_from_plain_text_passthrough():
    assert extract_code("no fences here") == "no fences here"


def test_extract_code_handles_truncated_block_with_no_closing_fence():
    # Regression guard: caught in a real run where a truncated generation
    # left "```python" as the literal first line of the "extracted" code,
    # guaranteeing a SyntaxError on execution. An opening fence with no
    # closing one must still yield the code after it, not the raw fence.
    llm_output = "```python\nx = 1\nprint(x)"
    result = extract_code(llm_output)
    assert result == "x = 1\nprint(x)"
    assert "```" not in result


def test_extract_code_from_json_corrected_code_key():
    payload = json.dumps({"corrected_code": "print('fixed')"})
    assert extract_code(payload) == "print('fixed')"


def test_extract_code_from_json_corrected_parts_list():
    payload = json.dumps({"corrected_parts": [{"corrected_code": "print('fixed2')"}]})
    assert extract_code(payload) == "print('fixed2')"


def test_run_code_safely_success():
    ok, output = run_code_safely("print(2 + 2)")
    assert ok is True
    assert output == "4"


def test_run_code_safely_syntax_error():
    ok, output = run_code_safely("def broken(:\n    pass")
    assert ok is False
    assert output


def test_run_code_safely_empty_stdout_is_failure():
    ok, output = run_code_safely("x = 1")
    assert ok is False
    assert output == "empty output"


def test_run_code_safely_timeout():
    ok, output = run_code_safely("import time; time.sleep(5)", timeout_s=0.2)
    assert ok is False
    assert "exceeded" in output
