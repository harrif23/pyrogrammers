import pytest

from evals.geometry import box_problem, normalized_to_pixels


def test_conversion_scales_each_axis_by_its_own_dimension():
    assert normalized_to_pixels((0.5, 0.25), 2560, 1440) == [1280.0, 360.0]
    assert normalized_to_pixels((0.0, 0.0), 2560, 1440) == [0.0, 0.0]
    assert normalized_to_pixels((1.0, 1.0), 2560, 1440) == [2560.0, 1440.0]


@pytest.mark.parametrize("bbox", [(5, 1, 1, 5), (1, 5, 5, 1), (1, 1, 21, 5), (1, 1, 5, 11),
                                  (-1, 1, 5, 5), (3, 1, 3, 5)])
def test_invalid_boxes_are_named(bbox):
    assert box_problem(bbox, 20, 10) is not None


def test_valid_box_has_no_problem():
    assert box_problem((0, 0, 20, 10), 20, 10) is None
