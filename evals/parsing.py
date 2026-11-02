"""Strict parsing of the model's final answer into a normalized point."""

import json
import math
import re
from dataclasses import dataclass

_THINK_BLOCK = re.compile(r"\A\s*<think>.*?</think>", re.DOTALL)


@dataclass(frozen=True)
class ParsedAnswer:
    point: tuple[float, float] | None
    invalid_reason: str | None


def _invalid(reason):
    return ParsedAnswer(None, reason)


def _reject_constant(name):
    raise ValueError(f"non-finite constant {name}")


def _reject_duplicate_keys(pairs):
    keys = [key for key, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate keys")
    return dict(pairs)


def parse_point(content):
    """Parse exactly {"x": number, "y": number} with both values in [0, 1].

    rules: no coordinate-format guessing. Pixels, fences, prose, booleans, extra keys and
    out-of-range values are all invalid, because recovering them would credit the model
    for output it was told not to produce.
    why: a complete leading <think> block is removed because some deployments return
    reasoning inline; it is not part of the answer, and the raw text is still logged.
    """
    if content is None or not content.strip():
        return _invalid("empty")
    text = _THINK_BLOCK.sub("", content, count=1).strip()
    if not text:
        return _invalid("empty_after_reasoning")
    try:
        data = json.loads(text, parse_constant=_reject_constant,
                          object_pairs_hook=_reject_duplicate_keys)
    except ValueError:
        return _invalid("not_json")
    if not isinstance(data, dict):
        return _invalid("not_object")
    if set(data) != {"x", "y"}:
        return _invalid("wrong_keys")

    coordinates = []
    for key in ("x", "y"):
        value = data[key]
        # bool is a subclass of int, so isinstance alone would accept true as 1.
        if type(value) not in (int, float):
            return _invalid("non_numeric")
        if isinstance(value, float) and not math.isfinite(value):
            return _invalid("non_finite")
        if not 0 <= value <= 1:
            return _invalid("out_of_range")
        coordinates.append(float(value))
    return ParsedAnswer((coordinates[0], coordinates[1]), None)
