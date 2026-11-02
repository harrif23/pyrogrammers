import pytest

from evals.paper_scorer import (calc_metric_for_result_list, eval_sample_positive_gt,
                                   score_prediction)

SAMPLE = {"bbox": [10, 20, 20, 30], "img_size": [100, 100]}


@pytest.mark.parametrize("point,expected", [
    ([0.15, 0.25], "correct"),
    ([0.10, 0.25], "correct"),   # left edge
    ([0.20, 0.25], "correct"),   # right edge
    ([0.15, 0.20], "correct"),   # top edge
    ([0.15, 0.30], "correct"),   # bottom edge
    ([0.10, 0.20], "correct"),   # corner
    ([0.0999, 0.25], "wrong"),
    ([0.2001, 0.25], "wrong"),
    ([0.15, 0.1999], "wrong"),
    ([0.15, 0.3001], "wrong"),
    (None, "wrong_format"),
])
def test_outcomes_match_the_published_scorer(point, expected):
    assert eval_sample_positive_gt(SAMPLE, {"point": point}) == expected


def test_box_is_scaled_per_axis_by_image_size():
    sample = {"bbox": [1000, 10, 2000, 20], "img_size": [2000, 100]}
    assert eval_sample_positive_gt(sample, {"point": [0.75, 0.15]}) == "correct"
    assert eval_sample_positive_gt(sample, {"point": [0.15, 0.75]}) == "wrong"
    # Above the box's top edge at 10/100; scaling that edge by the width would admit this point.
    assert eval_sample_positive_gt(sample, {"point": [0.75, 0.05]}) == "wrong"


def test_bbox_is_corners_not_width_height():
    # Under x, y, w, h this box would reach x=1211 and exclude the point.
    sample = {"bbox": [1201, 1032, 1230, 1060], "img_size": [2560, 1440]}
    assert eval_sample_positive_gt(sample, {"point": [1220 / 2560, 1050 / 1440]}) == "correct"


def test_metric_keys_and_denominators_match_the_published_code():
    results = [{"correctness": "correct", "ui_type": "icon"},
               {"correctness": "wrong", "ui_type": "icon"},
               {"correctness": "wrong_format", "ui_type": "text"},
               {"correctness": "correct", "ui_type": "text"}]
    assert calc_metric_for_result_list(results) == {
        "num_correct_action": 2, "num_total": 4, "wrong_format_num": 1,
        "action_acc": 0.5, "text_acc": 0.5, "icon_acc": 0.5}


def test_empty_metrics_are_zero_like_the_published_code():
    metrics = calc_metric_for_result_list([])
    assert metrics["action_acc"] == 0 and metrics["text_acc"] == 0 and metrics["icon_acc"] == 0


def test_score_prefers_the_model_normalized_point():
    # The stored normalized point decides; the pixel point here is deliberately elsewhere.
    assert score_prediction([0.15, 0.25], [99.0, 99.0], [10, 20, 20, 30], 100, 100) == "correct"


def test_score_normalizes_a_pixel_only_point():
    assert score_prediction(None, [15.0, 25.0], [10, 20, 20, 30], 100, 100) == "correct"
    assert score_prediction(None, [50.0, 25.0], [10, 20, 20, 30], 100, 100) == "wrong"


def test_score_without_a_point_is_wrong_format():
    assert score_prediction(None, None, [10, 20, 20, 30], 100, 100) == "wrong_format"
