"""Runs the repo's FLGround harness under the shared scorer: --predictor evals.flground_adapter:baseline"""

import os
import time

import openai
from PIL import Image
from pydantic import ValidationError

from flground.harness import ground
from flground.model import FireworksModel

from .baseline import make_prediction
from .client import failure_from_api_error


class FLGroundPredictor:
    """Adapts flground.harness.ground to predict(image, instruction).

    why: the harness keeps its own prompt and model settings, so its score measures the harness
    as written. ground() raises on bad output and drops earlier calls, so a failed attempt
    logs no tokens and no raw answer.
    """

    def __init__(self, strategy):
        self._strategy = strategy
        self._model = FireworksModel()

    def predict(self, image, instruction):
        started = time.perf_counter()
        with Image.open(image) as opened:
            opened.load()
            try:
                result = ground(opened, instruction, self._model, strategy=self._strategy)
            except openai.APIError as error:
                elapsed = time.perf_counter() - started
                key = os.environ.get("FIREWORKS_API_KEY", "")
                failure = failure_from_api_error(error, elapsed, key)
                return make_prediction(latency_seconds=elapsed, error=failure.to_dict())
            except (ValidationError, ValueError) as error:
                return make_prediction(latency_seconds=time.perf_counter() - started,
                                       invalid_reason=type(error).__name__)
        return make_prediction(
            point_px=[result.point.x, result.point.y],
            calls=[{"content": call.raw_response, "reasoning_content": None, "finish_reason": None,
                    "response_id": None, "returned_model": result.model,
                    "latency_seconds": call.latency_seconds, "usage": call.usage or None}
                   for call in result.calls],
            latency_seconds=result.latency_seconds,
        )


def baseline(client):
    return FLGroundPredictor("baseline")


def refine(client):
    return FLGroundPredictor("refine")
