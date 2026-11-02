"""The predict interface and the model-only baseline: full screenshot to original-image point."""

import hashlib
import time
from pathlib import Path
from typing import Protocol, TypedDict

from PIL import Image

from .client import FireworksClient, encode_image
from .errors import ApiFailure
from .geometry import normalized_to_pixels
from .parsing import parse_point

PROMPT_TEMPLATE = (
    "Locate the user interface element described by this instruction in the screenshot.\n"
    "Instruction: {instruction}\n\n"
    'Respond with only a JSON object of the form {{"x": number, "y": number}} giving the '
    "center of that element, normalized to the full image: x=0 is the left edge, x=1 the "
    "right edge, y=0 the top edge, y=1 the bottom edge. Output nothing else."
)
PROMPT_SHA256 = hashlib.sha256(PROMPT_TEMPLATE.encode()).hexdigest()


class PredictionResult(TypedDict):
    point_px: list[float] | None
    point_normalized: list[float] | None
    calls: list[dict]
    latency_seconds: float
    fallback_used: bool
    invalid_reason: str | None
    error: dict | None


def make_prediction(point_px=None, point_normalized=None, calls=None, latency_seconds=0.0,
                    fallback_used=False, invalid_reason=None, error=None):
    """Build a valid predict() result. Harnesses should use this rather than a hand-made dict."""
    return PredictionResult(
        point_px=point_px, point_normalized=point_normalized, calls=calls or [],
        latency_seconds=latency_seconds, fallback_used=fallback_used,
        invalid_reason=invalid_reason, error=error,
    )


class Predictor(Protocol):
    """Anything that points at a UI element. Crop-and-refine will implement this too.

    why: the signature has no room for gold boxes or labels, so a predictor cannot see
    them, and every strategy is judged by the same scorer.
    """

    def predict(self, image: Path, instruction: str) -> PredictionResult: ...


class BaselinePredictor:
    """Sends the whole original screenshot to the model once."""

    def __init__(self, client: FireworksClient):
        self._client = client

    def predict(self, image, instruction):
        """Predict one point.

        rules: API failures are returned in `error` rather than raised, so a multi-call
        strategy keeps the tokens of calls that already succeeded.
        side effects: one model call.
        raises: ImageInputError for an image that cannot be sent unmodified.
        """
        started = time.perf_counter()
        image_uri = encode_image(image)
        with Image.open(image) as opened:
            width, height = opened.size
        prompt = PROMPT_TEMPLATE.format(instruction=instruction)
        try:
            call = self._client.complete(image_uri, prompt)
        except ApiFailure as failure:
            return PredictionResult(
                point_px=None, point_normalized=None, calls=[],
                latency_seconds=failure.latency_seconds or time.perf_counter() - started,
                fallback_used=False, invalid_reason=None, error=failure.to_dict(),
            )
        parsed = parse_point(call["content"])
        return PredictionResult(
            point_px=normalized_to_pixels(parsed.point, width, height) if parsed.point else None,
            point_normalized=list(parsed.point) if parsed.point else None,
            calls=[call],
            latency_seconds=call["latency_seconds"],
            fallback_used=False,
            invalid_reason=parsed.invalid_reason,
            error=None,
        )
