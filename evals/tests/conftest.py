import json

import httpx
import pytest
from PIL import Image

from evals.baseline import BaselinePredictor
from evals.client import FireworksClient
from evals.config import Credentials, build_inference_config
from evals.dataset import Example

API_KEY = "test-key-not-real-abc123"
MODEL = "accounts/acme/deployments/dep1"


def completion_body(content='{"x": 0.5, "y": 0.5}', finish_reason="stop", usage="default"):
    body = {
        "id": "resp-1",
        "object": "chat.completion",
        "created": 1,
        "model": MODEL,
        "choices": [{"index": 0, "finish_reason": finish_reason,
                     "message": {"role": "assistant", "content": content,
                                 "reasoning_content": "thinking"}}],
    }
    if usage == "default":
        body["usage"] = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
                         "prompt_tokens_details": {"cached_tokens": 10},
                         "completion_tokens_details": {"reasoning_tokens": 12}}
    elif usage is not None:
        body["usage"] = usage
    return body


class FakeServer:
    """Serves queued responses through the real OpenAI client, recording what went on the wire."""

    def __init__(self):
        self.responses = []
        self.requests = []

    def queue(self, *responses):
        self.responses.extend(responses)

    def __call__(self, request):
        self.requests.append(request)
        status, body = self.responses.pop(0) if self.responses else (200, completion_body())
        return httpx.Response(status, json=body)

    def bodies(self):
        return [json.loads(request.content) for request in self.requests]


@pytest.fixture
def server():
    return FakeServer()


@pytest.fixture
def make_client(server):
    def build(max_retries=0, **settings):
        return FireworksClient(Credentials(API_KEY, MODEL), build_inference_config(**settings),
                               http_client=httpx.Client(transport=httpx.MockTransport(server)),
                               max_retries=max_retries)
    return build


@pytest.fixture
def make_predictor(make_client):
    def build(**settings):
        return BaselinePredictor(make_client(**settings))
    return build


@pytest.fixture
def image_path(tmp_path):
    path = tmp_path / "screen.png"
    Image.new("RGB", (200, 100), (10, 20, 30)).save(path)
    return path


@pytest.fixture
def make_examples(image_path):
    def build(*ui_types):
        return [Example(id=f"ex_{index}", image_path=image_path,
                        instruction=f"click item {index}", bbox=(90, 40, 110, 60),
                        width=200, height=100, ui_type=ui_type)
                for index, ui_type in enumerate(ui_types)]
    return build


@pytest.fixture
def meta():
    return {"mode": "baseline", "model": MODEL, "inference": {"max_tokens": 4096},
            "fingerprint": "fp"}
