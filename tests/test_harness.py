import pytest
from PIL import Image
from pydantic import ValidationError
from flground.contracts import Point, CallRecord
from flground.geometry import Transform, crop_bounds, validate_point
from flground.harness import ground


class FakeModel:
    model = "fake"
    def __init__(self, points):
        self.points = iter(points)
        self.sizes = []
    def predict(self, image, instruction):
        self.sizes.append(image.size)
        return CallRecord(point=next(self.points), raw_response="{}", latency_seconds=0)


def test_refinement_maps_crop_back_and_limits_calls():
    model = FakeModel([Point(x=900, y=700), Point(x=200, y=100)])
    result = ground(Image.new("RGB", (1000, 800)), "volume", model, "refine", crop_size=200)
    assert result.point == Point(x=900, y=650)
    assert model.sizes == [(1000, 800), (400, 400)]
    assert result.model_calls == 2


def test_baseline_one_call():
    model = FakeModel([Point(x=5, y=7)])
    result = ground(Image.new("RGB", (10, 10)), "volume", model)
    assert result.point == Point(x=5, y=7)
    assert result.model_calls == 1


def test_crop_at_edge():
    assert crop_bounds(Point(x=0, y=0), 100, 80, 200) == (0, 0, 100, 80)


def test_retina_mapping():
    assert Transform(0, 0, 1440, 900, 2880, 1800).to_source(Point(x=1000, y=600)) == Point(x=500, y=300)


@pytest.mark.parametrize("x,y", [(-1, 0), (100, 0), (0, 80)])
def test_out_of_bounds_rejected(x, y):
    with pytest.raises(ValueError):
        validate_point(Point(x=x, y=y), 100, 80)


def test_nonfinite_rejected():
    with pytest.raises(ValidationError):
        Point(x=float("nan"), y=2)


def test_malformed_response_rejected():
    with pytest.raises(ValidationError):
        Point.model_validate_json('{"x": 1}')
