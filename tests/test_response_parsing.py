import pytest

from flground.contracts import Point
from flground.geometry import validate_point
from flground.model import parse_point


@pytest.mark.parametrize('raw', [
    '{"x": 3.5, "y": 4}',
    '```json\n{"x": 3.5, "y": 4}\n```',
    '```\n[{"point_2d": [3.5, 4]}]\n```',
    '{"point_2d": [3.5, 4], "label": "volume"}',
])
def test_single_target_formats(raw):
    assert parse_point(raw) == Point(x=3.5, y=4)


@pytest.mark.parametrize('raw', [
    '', 'not json', '[]',
    '[{"point_2d": [1, 2]}, {"point_2d": [3, 4]}]',
    '{"point_2d": [1]}', '{"point_2d": [1, 2, 3]}',
    '{"point_2d": [1, 2], "x": 3, "y": 4}',
    '{"x": true, "y": 2}', '{"x": "1", "y": 2}',
    '{"x": NaN, "y": 2}', '{"x": 1}',
    'Here is the target: {"x": 1, "y": 2}',
])
def test_invalid_or_ambiguous_responses_rejected(raw):
    with pytest.raises(ValueError):
        parse_point(raw)


@pytest.mark.parametrize('coordinates', ['[-1, 0]', '[100, 0]', '[0, 80]'])
def test_alternate_format_preserves_bounds_validation(coordinates):
    point = parse_point('{"point_2d": ' + coordinates + '}')
    with pytest.raises(ValueError):
        validate_point(point, 100, 80)
