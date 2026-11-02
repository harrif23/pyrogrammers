import base64

import pytest

from conftest import API_KEY, MODEL, completion_body
from evals import client as client_module
from evals.client import encode_image
from evals.errors import ApiFailure, ImageInputError


def test_request_on_the_wire(server, make_client):
    make_client(reasoning_effort="medium", max_tokens=1234).complete("data:image/png;base64,AA", "hi")
    body = server.bodies()[0]
    assert body["model"] == MODEL
    assert body["max_tokens"] == 1234
    assert body["temperature"] == 0.0
    assert body["reasoning_effort"] == "medium"
    assert body["context_length_exceeded_behavior"] == "error"
    assert "response_format" not in body
    image_part, text_part = body["messages"][0]["content"]
    assert image_part == {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}
    assert text_part == {"type": "text", "text": "hi"}
    assert str(server.requests[0].url).startswith("https://api.fireworks.ai/inference/v1/")
    assert server.requests[0].headers["authorization"] == f"Bearer {API_KEY}"


def test_reasoning_can_be_omitted_or_capped(server, make_client):
    make_client(reasoning_effort="omit").complete("uri", "x")
    make_client(reasoning_effort="500").complete("uri", "x")
    assert "reasoning_effort" not in server.bodies()[0]
    assert server.bodies()[1]["reasoning_effort"] == 500


def test_call_record_captures_response_metadata(server, make_client):
    call = make_client().complete("uri", "x")
    assert call["content"] == '{"x": 0.5, "y": 0.5}'
    assert call["reasoning_content"] == "thinking"
    assert call["finish_reason"] == "stop"
    assert call["response_id"] == "resp-1"
    assert call["returned_model"] == MODEL
    assert call["latency_seconds"] >= 0
    assert call["usage"]["completion_tokens_details"]["reasoning_tokens"] == 12


def test_absent_usage_stays_none(server, make_client):
    server.queue((200, completion_body(usage=None)))
    assert make_client().complete("uri", "x")["usage"] is None


def test_empty_choices_is_an_api_failure(server, make_client):
    body = completion_body()
    body["choices"] = []
    server.queue((200, body))
    with pytest.raises(ApiFailure) as caught:
        make_client().complete("uri", "x")
    assert caught.value.kind == "malformed_response"


@pytest.mark.parametrize("status,kind,systematic", [
    (401, "authentication", True),
    (403, "permission", True),
    (404, "deployment_not_found", True),
    (429, "api_status", False),
    (500, "api_status", False),
    (503, "api_status", False),
])
def test_api_errors_are_classified_without_retry(server, make_client, status, kind, systematic):
    server.queue((status, {"error": {"message": "nope"}}))
    with pytest.raises(ApiFailure) as caught:
        make_client().complete("uri", "x")
    assert caught.value.kind == kind
    assert caught.value.status == status
    assert caught.value.systematic is systematic
    assert len(server.requests) == 1


def test_retry_policy_is_explicit(server, make_client):
    server.queue((500, {"error": {"message": "boom"}}), (200, completion_body()))
    assert make_client(max_retries=1).complete("uri", "x")["finish_reason"] == "stop"
    assert len(server.requests) == 2


def test_api_key_is_redacted_from_error_messages(server, make_client):
    server.queue((400, {"error": {"message": f"bad key {API_KEY}"}}))
    with pytest.raises(ApiFailure) as caught:
        make_client().complete("uri", "x")
    assert API_KEY not in caught.value.message
    assert "[REDACTED]" in caught.value.message


def test_image_is_encoded_byte_for_byte(image_path):
    prefix, encoded = encode_image(image_path).split(",", 1)
    assert prefix == "data:image/png;base64"
    assert base64.b64decode(encoded) == image_path.read_bytes()


def test_oversized_image_is_refused_rather_than_resized(image_path, monkeypatch):
    monkeypatch.setattr(client_module, "MAX_BASE64_BYTES", 10)
    with pytest.raises(ImageInputError):
        encode_image(image_path)
