import base64
import io
import inspect
import json

import pytest
from PIL import Image

from conftest import completion_body
from evals import cli, evaluate, reference_qwen
from evals.errors import EXIT_OK, ImageInputError
from evals.reference_qwen import (SYSTEM_PROMPT, ReferenceQwenPredictor, encode_reference_image,
                                     parse_tool_call, smart_resize)
from workspace_fixture import workspace  # noqa: F401


def tool_call(coordinate):
    arguments = {"action": "left_click", "coordinate": coordinate}
    return ('<tool_call>\n{"name": "computer_use", "arguments": ' + json.dumps(arguments)
            + "}\n</tool_call>")


@pytest.mark.parametrize("response,point", [
    (tool_call([500, 250]), [0.5, 0.25]),
    (tool_call([0, 1000]), [0.0, 1.0]),
    (tool_call([100, 200, 300, 400]), [0.2, 0.3]),   # a box becomes its center
    (tool_call([1500, 250]), [1.5, 0.25]),           # no range check, as in the source
    ("thinking first\n" + tool_call([500, 250]), [0.5, 0.25]),
    (tool_call([1, 1]) + "\n" + tool_call([500, 250]), [0.5, 0.25]),  # the last call is used
])
def test_parse_accepts_what_the_authors_code_accepts(response, point):
    assert parse_tool_call(response) == (pytest.approx(point), None)


@pytest.mark.parametrize("response", [
    '{"x": 0.5, "y": 0.5}',
    "<tool_call>\nnot json\n</tool_call>",
    '<tool_call>{"arguments": {"coordinate": [1, 2]}}</tool_call>',    # no newlines
    '<tool_call>\n{"arguments": {"coordinate": [1, 2]}}',              # truncated
    '<tool_call>\n{"arguments": {}}\n</tool_call>',
    '<tool_call>\n{"arguments": {"coordinate": [1, 2, 3]}}\n</tool_call>',
    '<tool_call>\n{"arguments": {"coordinate": ["1", "2"]}}\n</tool_call>',
    '<tool_call>\n[1, 2]\n</tool_call>',
    "",
])
def test_parse_rejects_what_the_authors_code_rejects(response):
    assert parse_tool_call(response) == (None, "tool_call_parse_failed")


def test_empty_content_is_invalid():
    assert parse_tool_call(None) == (None, "empty")


@pytest.mark.parametrize("response", [
    '<tool_call>\n{"arguments": {"coordinate": [NaN, 2]}}\n</tool_call>',
    '<tool_call>\n{"arguments": {"coordinate": [Infinity, 2]}}\n</tool_call>',
])
def test_non_finite_values_are_invalid_and_can_be_logged(response):
    assert parse_tool_call(response) == (None, "non_finite")


def test_huge_integer_does_not_crash():
    response = '<tool_call>\n{"arguments": {"coordinate": [1' + "0" * 400 + ', 2]}}\n</tool_call>'
    assert parse_tool_call(response)[0] is None


def test_prompt_states_the_1000_scale_and_keeps_real_newlines():
    assert "The screen's resolution is 1000x1000." in SYSTEM_PROMPT
    assert "guess a most likely one.\n\n# Tools" in SYSTEM_PROMPT
    assert "\\n" not in SYSTEM_PROMPT


@pytest.mark.parametrize("height,width,expected", [
    (1440, 2560, (1440, 2560)),   # the size of every target in this benchmark's splits
    (1441, 2561, (1440, 2560)),   # rounds to the factor of 8
    (1440, 5120, (1440, 5120)),   # 7.37M pixels, under the limit
    (2160, 3840, (2160, 3840)),   # exactly the limit
    (4320, 7680, (2160, 3840)),   # over the limit, scaled down
    (10, 10, (32, 32)),           # under the minimum, scaled up
])
def test_smart_resize_matches_the_transformers_function(height, width, expected):
    assert smart_resize(height, width) == expected


def test_extreme_aspect_ratio_is_refused():
    with pytest.raises(ImageInputError):
        smart_resize(1, 300)


def decoded(uri):
    prefix, encoded = uri.split(",", 1)
    assert prefix == "data:image/png;base64"
    return Image.open(io.BytesIO(base64.b64decode(encoded)))


def test_image_is_rgb_png_at_the_same_size_when_no_resize_is_needed(tmp_path):
    path = tmp_path / "a.png"
    Image.new("RGBA", (2560, 1440), (9, 8, 7, 255)).save(path)
    uri, width, height = encode_reference_image(path)
    image = decoded(uri)
    assert (width, height) == (2560, 1440)
    assert image.size == (2560, 1440)
    assert image.mode == "RGB"
    assert image.getpixel((5, 5)) == (9, 8, 7)


def test_oversized_image_is_resized_but_reports_original_dimensions(tmp_path):
    path = tmp_path / "a.png"
    Image.new("RGB", (7680, 4320)).save(path)
    uri, width, height = encode_reference_image(path)
    assert decoded(uri).size == (3840, 2160)
    assert (width, height) == (7680, 4320)


def test_oversized_payload_is_refused(image_path, monkeypatch):
    monkeypatch.setattr(reference_qwen, "MAX_BASE64_BYTES", 10)
    with pytest.raises(ImageInputError):
        encode_reference_image(image_path)


def test_request_has_the_system_prompt_then_image_then_bare_instruction(server, make_client, image_path):
    ReferenceQwenPredictor(make_client(reasoning_effort="xhigh")).predict(image_path, "turn off AMP")
    body = server.bodies()[0]
    system, user = body["messages"]
    assert system == {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}
    assert user["role"] == "user"
    assert user["content"][0]["type"] == "image_url"
    assert user["content"][1] == {"type": "text", "text": "turn off AMP"}
    assert body["reasoning_effort"] == "xhigh"
    assert body["max_tokens"] == 4096
    assert body["temperature"] == 0.0
    assert "tools" not in body


def test_predict_signature_cannot_receive_gold_data():
    parameters = inspect.signature(ReferenceQwenPredictor.predict).parameters
    assert list(parameters) == ["self", "image", "instruction"]


def test_prediction_maps_the_1000_scale_to_original_pixels(server, make_client, image_path):
    server.queue((200, completion_body(content=tool_call([500, 250]))))
    prediction = ReferenceQwenPredictor(make_client()).predict(image_path, "x")
    assert prediction["point_normalized"] == [0.5, 0.25]
    assert prediction["point_px"] == [100.0, 25.0]
    assert prediction["invalid_reason"] is None
    assert prediction["fallback_used"] is False


def test_unparsed_answer_is_invalid_not_an_error(server, make_client, image_path):
    server.queue((200, completion_body(content='{"x": 0.5, "y": 0.5}')))
    prediction = ReferenceQwenPredictor(make_client()).predict(image_path, "x")
    assert prediction["point_px"] is None
    assert prediction["invalid_reason"] == "tool_call_parse_failed"
    assert prediction["error"] is None


def test_api_failure_is_returned_in_error(server, make_client, image_path):
    server.queue((503, {"error": {"message": "nope"}}))
    prediction = ReferenceQwenPredictor(make_client()).predict(image_path, "x")
    assert prediction["error"]["status"] == 503
    assert prediction["calls"] == []


def test_offscreen_point_scores_wrong_not_wrong_format(server, make_client, make_examples, meta, tmp_path):
    results = tmp_path / "r.jsonl"
    server.queue((200, completion_body(content=tool_call([1500, 500]))))
    evaluate.evaluate_examples(make_examples("icon"), ReferenceQwenPredictor(make_client()), results, meta)
    record = json.loads(results.read_text())
    assert record["correctness"] == "wrong"


def test_label_is_kept_in_the_config_and_report_and_cannot_change_on_resume(workspace, server):
    label = "Qwen3.8-27B on the FL subset using the Qwen3.5 adapter protocol."
    out = workspace / "outputs" / "labeled.jsonl"
    server.queue((401, {"error": {"message": "bad"}}))
    assert cli.main(["run", "--mode", "reference", "--label", label, "--out", str(out)]) != EXIT_OK
    assert cli.main(["run", "--resume", "--label", "other", "--out", str(out)]) != EXIT_OK
    server.queue(*[(200, completion_body(content=tool_call([500, 500])))] * 8)
    assert cli.main(["run", "--resume", "--out", str(out)]) == EXIT_OK
    assert json.loads(out.with_suffix(".report.json").read_text())["label"] == label
    assert json.loads(out.with_suffix(".config.json").read_text())["label"] == label


def test_the_default_run_is_the_qwen35_protocol_at_xhigh_on_dev(workspace, server):
    out = workspace / "outputs" / "default.jsonl"
    server.queue((200, completion_body(content=tool_call([500, 500]))))
    assert cli.main(["run", "--limit", "1", "--out", str(out)]) == EXIT_OK
    written = json.loads(out.with_suffix(".report.json").read_text())
    assert written["mode"] == "reference"
    assert written["split"] == "dev"
    body = server.bodies()[0]
    assert body["reasoning_effort"] == "xhigh"
    assert body["max_tokens"] == 4096
    assert body["messages"][0]["role"] == "system"


def test_the_two_modes_have_different_prompt_hashes():
    assert evaluate.PROMPT_SHA256["simple_json"] != evaluate.PROMPT_SHA256["reference"]


def test_reference_mode_end_to_end_and_resume_keeps_its_mode(workspace, server):
    out = workspace / "outputs" / "reference.jsonl"
    server.queue((401, {"error": {"message": "bad"}}))
    code = cli.main(["run", "--mode", "reference", "--reasoning-effort", "xhigh", "--split", "eval", "--out", str(out)])
    assert code != EXIT_OK
    server.queue(*[(200, completion_body(content=tool_call([500, 500])))] * 3)
    assert cli.main(["run", "--out", str(out), "--resume"]) == EXIT_OK
    written = json.loads(out.with_suffix(".report.json").read_text())
    assert written["mode"] == "reference"
    assert written["overall"] == {"correct": 3, "total": 3, "accuracy": 1.0}
    assert written["finalized"] is True
    assert "system" in [m["role"] for m in server.bodies()[-1]["messages"]]


def test_reference_and_baseline_runs_compare(workspace, server, capsys):
    a, b = [str(workspace / f"outputs/{name}.jsonl") for name in ("a", "b")]
    assert cli.main(["run", "--mode", "simple_json", "--split", "eval", "--out", a]) == EXIT_OK
    server.queue(*[(200, completion_body(content=tool_call([1, 1])))] * 3)
    assert cli.main(["run", "--split", "eval", "--mode", "reference", "--out", b]) == EXIT_OK
    capsys.readouterr()
    assert cli.main(["report", a, b]) == EXIT_OK
    assert json.loads(capsys.readouterr().out)["a_only"] == 3
