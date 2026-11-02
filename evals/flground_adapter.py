"""Runs the repo's FLGround strategies under the shared scorer: --predictor evals.flground_adapter:panel"""

import os
import time

import openai
from PIL import Image
from pydantic import ValidationError

from flground.harness import ground
from flground.model import FireworksModel

from .baseline import make_prediction
from .client import failure_from_api_error


def _call_record(call, model):
    return {"content": call.raw_response, "reasoning_content": None,
            "finish_reason": call.finish_reason, "response_id": None, "returned_model": model,
            "latency_seconds": call.latency_seconds, "usage": call.usage or None}


class FLGroundPredictor:
    """Adapts flground.harness.ground to predict(image, instruction).

    why: strategies keep their own prompts and model settings, so a score measures the strategy
    as written. Calls are read from the model's own log, because ground() raises on bad output
    and a failed example must still keep the tokens and raw answers of the calls it made.
    """

    def __init__(self, strategy):
        self._strategy = strategy
        self._model = FireworksModel()

    def predict(self, image, instruction):
        started = time.perf_counter()
        first_call = len(self._model.records)
        point, invalid_reason, error = None, None, None
        with Image.open(image) as opened:
            rgb = opened.convert("RGB")
        try:
            result = ground(rgb, instruction, self._model, strategy=self._strategy)
        except openai.APIError as failure:
            key = os.environ.get("FIREWORKS_API_KEY", "")
            error = failure_from_api_error(failure, time.perf_counter() - started, key).to_dict()
        except (ValidationError, ValueError) as failure:
            invalid_reason = type(failure).__name__
        else:
            if result.point is None:
                # The model reported the target absent, which is a miss on a benchmark target.
                invalid_reason = result.status
            else:
                point = [result.point.x, result.point.y]
        calls = [_call_record(call, self._model.model) for call in self._model.records[first_call:]]
        return make_prediction(point_px=point, calls=calls, invalid_reason=invalid_reason, error=error,
                               latency_seconds=time.perf_counter() - started)


def baseline(client):
    return FLGroundPredictor("baseline")


def knowledge(client):
    return FLGroundPredictor("knowledge")


def refine(client):
    return FLGroundPredictor("refine")


def panel_plain(client):
    return FLGroundPredictor("panel_plain")


def panel(client):
    return FLGroundPredictor("panel")


def panel_verify(client):
    return FLGroundPredictor("panel_verify")
