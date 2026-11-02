"""The ScreenSpot-Pro scorer and metrics, ported so published and local numbers mean the same thing.

Source: likaixin2000/ScreenSpot-Pro-GUI-Grounding, eval_screenspot_pro.py, commit
6ca3cf9079836452ef947ccd68a16393116d3405: eval_sample_positive_gt, calc_metric_for_result_list.
Function names, return values and metric keys match the source; keep them in sync with it.
"""


def eval_sample_positive_gt(sample, response):
    """Compare a normalized point to a gold [x1, y1, x2, y2] box scaled by the image size.

    returns: "correct", "wrong", or "wrong_format" when there is no point.
    """
    bbox = sample["bbox"]
    img_size = sample["img_size"]
    bbox = [bbox[0] / img_size[0], bbox[1] / img_size[1], bbox[2] / img_size[0], bbox[3] / img_size[1]]

    click_point = response["point"]
    if click_point is None:
        return "wrong_format"
    if (bbox[0] <= click_point[0] <= bbox[2]) and (bbox[1] <= click_point[1] <= bbox[3]):
        return "correct"
    return "wrong"


def calc_metric_for_result_list(results):
    """Metrics for a result list. wrong_format results count in every denominator."""
    num_total = len(results)
    correct_num = sum(1 for res in results if res["correctness"] == "correct")
    wrong_format_num = sum(1 for res in results if res["correctness"] == "wrong_format")

    text_results = [res for res in results if res.get("ui_type") == "text"]
    icon_results = [res for res in results if res.get("ui_type") == "icon"]
    text_correct = sum(1 for res in text_results if res["correctness"] == "correct")
    icon_correct = sum(1 for res in icon_results if res["correctness"] == "correct")
    return {
        "num_correct_action": correct_num,
        "num_total": num_total,
        "wrong_format_num": wrong_format_num,
        "action_acc": correct_num / num_total if num_total > 0 else 0,
        "text_acc": text_correct / len(text_results) if text_results else 0,
        "icon_acc": icon_correct / len(icon_results) if icon_results else 0,
    }


def score_prediction(point_normalized, point_px, bbox, width, height):
    """Score one prediction, normalizing a pixel point for strategies that only return pixels."""
    point = point_normalized
    if point is None and point_px is not None:
        point = [point_px[0] / width, point_px[1] / height]
    return eval_sample_positive_gt({"bbox": bbox, "img_size": [width, height]}, {"point": point})
