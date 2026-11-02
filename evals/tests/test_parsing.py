import pytest

from evals.parsing import parse_point


@pytest.mark.parametrize("text,point", [
    ('{"x": 0.25, "y": 0.75}', (0.25, 0.75)),
    ('  {"y": 0.75, "x": 0.25}\n', (0.25, 0.75)),
    ('{"x": 0, "y": 1}', (0.0, 1.0)),
    ('<think>maybe {"x": 0.9, "y": 0.9}</think>\n{"x": 0.1, "y": 0.2}', (0.1, 0.2)),
])
def test_accepts_only_exact_normalized_json(text, point):
    assert parse_point(text).point == point


@pytest.mark.parametrize("text,reason", [
    (None, "empty"),
    ("   ", "empty"),
    ("<think>reasoning only</think>", "empty_after_reasoning"),
    ('<think>never closed {"x": 0.1, "y": 0.2}', "not_json"),
    ('```json\n{"x": 0.1, "y": 0.2}\n```', "not_json"),
    ('The answer is {"x": 0.1, "y": 0.2}', "not_json"),
    ('{"x": 0.1, "y": 0.2', "not_json"),
    ("[0.1, 0.2]", "not_object"),
    ('{"x": 0.1}', "wrong_keys"),
    ('{"x": 0.1, "y": 0.2, "z": 0.3}', "wrong_keys"),
    ('{"x": true, "y": 0.2}', "non_numeric"),
    ('{"x": "0.1", "y": 0.2}', "non_numeric"),
    ('{"x": null, "y": 0.2}', "non_numeric"),
    ('{"x": 1e999, "y": 0.2}', "non_finite"),
    ('{"x": NaN, "y": 0.2}', "not_json"),
    ('{"x": Infinity, "y": 0.2}', "not_json"),
    ('{"x": 1.0001, "y": 0.2}', "out_of_range"),
    ('{"x": -0.01, "y": 0.2}', "out_of_range"),
    ('{"x": 1280, "y": 720}', "out_of_range"),
    ('{"x": 0.1, "x": 0.2, "y": 0.3}', "not_json"),
])
def test_rejects_without_guessing(text, reason):
    parsed = parse_point(text)
    assert parsed.point is None
    assert parsed.invalid_reason == reason


def test_huge_integer_is_rejected_rather_than_overflowing():
    assert parse_point('{"x": 1' + "0" * 400 + ', "y": 0.2}').invalid_reason == "out_of_range"
