import base64
import hashlib
import inspect

import pytest

from conftest import completion_body
from evals.baseline import PROMPT_SHA256, PROMPT_TEMPLATE, BaselinePredictor


def test_returns_a_point_in_original_image_pixels(server, make_predictor, image_path):
    server.queue((200, completion_body(content='{"x": 0.25, "y": 0.5}')))
    prediction = make_predictor().predict(image_path, "x")
    assert prediction["point_px"] == [50.0, 50.0]
    assert prediction["point_normalized"] == [0.25, 0.5]
    assert prediction["error"] is None
    assert prediction["invalid_reason"] is None
    assert prediction["fallback_used"] is False


def test_prediction_has_exactly_the_agreed_fields(server, make_predictor, image_path):
    assert set(make_predictor().predict(image_path, "x")) == {
        "point_px", "point_normalized", "calls", "latency_seconds", "fallback_used",
        "invalid_reason", "error"}


def test_baseline_makes_one_call_and_records_it(server, make_predictor, image_path):
    prediction = make_predictor().predict(image_path, "x")
    assert len(server.requests) == 1
    assert len(prediction["calls"]) == 1
    assert prediction["latency_seconds"] == prediction["calls"][0]["latency_seconds"]


def test_sends_the_original_file_and_the_instruction(server, make_predictor, image_path):
    make_predictor().predict(image_path, "turn off AMP")
    image_part, text_part = server.bodies()[0]["messages"][0]["content"]
    encoded = image_part["image_url"]["url"].split(",", 1)[1]
    assert base64.b64decode(encoded) == image_path.read_bytes()
    assert text_part["text"] == PROMPT_TEMPLATE.format(instruction="turn off AMP")


def test_prompt_is_unchanged_from_the_live_dev_runs():
    # Changing the prompt must be a deliberate decision, never a side effect of a refactor.
    assert PROMPT_SHA256.startswith("196e5799")
    assert PROMPT_SHA256 == hashlib.sha256(PROMPT_TEMPLATE.encode()).hexdigest()


def test_predict_signature_cannot_receive_gold_data():
    parameters = inspect.signature(BaselinePredictor.predict).parameters
    assert list(parameters) == ["self", "image", "instruction"]


def test_malformed_output_is_invalid_not_an_error(server, make_predictor, image_path):
    server.queue((200, completion_body(content="the top left")))
    prediction = make_predictor().predict(image_path, "x")
    assert prediction["point_px"] is None
    assert prediction["invalid_reason"] == "not_json"
    assert prediction["error"] is None
    assert prediction["calls"][0]["content"] == "the top left"


def test_pixel_answer_is_invalid_not_rescaled(server, make_predictor, image_path):
    server.queue((200, completion_body(content='{"x": 100, "y": 50}')))
    prediction = make_predictor().predict(image_path, "x")
    assert prediction["point_px"] is None
    assert prediction["invalid_reason"] == "out_of_range"


@pytest.mark.parametrize("status,systematic", [(401, True), (503, False)])
def test_api_failure_is_returned_in_error(server, make_predictor, image_path, status, systematic):
    server.queue((status, {"error": {"message": "nope"}}))
    prediction = make_predictor().predict(image_path, "x")
    assert prediction["point_px"] is None
    assert prediction["calls"] == []
    assert prediction["error"]["status"] == status
    assert prediction["error"]["systematic"] is systematic
    assert prediction["latency_seconds"] >= 0
