"""The benchmark authors' Qwen 3.5 adapter protocol: prompt, 0-1000 coordinates, parsing, images.

Results from another model aren't a reproduction of Qwen3.5's score; see QWEN35_ADAPTER_PROTOCOL.md.

Source: likaixin2000/ScreenSpot-Pro-GUI-Grounding, models/qwen3_5.py, commit
83f3041fdb76848e29a6c76094495b2489a3b817, class Qwen3_5Model (the non-prefilled Transformers path).
Resize: transformers, models/qwen2_vl/image_processing_qwen2_vl.py, smart_resize.
Reasoning and the output limit are run settings, not part of this module.
"""

import base64
import hashlib
import io
import json
import math
import time

from PIL import Image

from .baseline import PredictionResult
from .client import MAX_BASE64_BYTES, FireworksClient
from .errors import ApiFailure, ImageInputError
from .geometry import normalized_to_pixels

SYSTEM_PROMPT = """You are a helpful assistant. The user will give you an instruction, and you MUST left click on the corresponding UI element via tool call. If you are not sure about where to click, guess a most likely one.\n\n# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "computer_use", "description": "Use a mouse to interact with a computer.\n* The screen's resolution is 1000x1000.\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. \n* You can only use the left_click action to interact with the computer.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\n* `left_click`: Click the left mouse button with coordinate (x, y).", "enum": ["left_click"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=left_click`.", "type": "array"}, "required": ["action"], "type": "object"}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""
PROMPT_SHA256 = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()

MAX_PIXELS = 8294400  # QWEN3_5_MAX_PIXELS default in the Hugging Face path
_RESIZE_FACTOR = 8
_MIN_PIXELS = 32 * 32
_COORDINATE_SCALE = 1000


def smart_resize(height, width, factor=_RESIZE_FACTOR, min_pixels=_MIN_PIXELS, max_pixels=MAX_PIXELS):
    """Dimensions divisible by factor, within the pixel limits, keeping the aspect ratio close."""
    if max(height, width) / min(height, width) > 200:
        raise ImageInputError("absolute aspect ratio must be smaller than 200")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def encode_reference_image(image):
    """Convert to RGB, apply smart_resize, and PNG-encode, as the authors' code does.

    returns: (data URI, original width, original height). Coordinates stay relative to the
    whole image, so answers map back to the original size even when the image is resized.
    raises: ImageInputError if the encoded image exceeds the request limit.
    """
    with Image.open(image) as opened:
        rgb = opened.convert("RGB")
    width, height = rgb.size
    resized_height, resized_width = smart_resize(height, width)
    if (resized_width, resized_height) != (width, height):
        rgb = rgb.resize((resized_width, resized_height))
    buffer = io.BytesIO()
    rgb.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    if len(encoded) >= MAX_BASE64_BYTES:
        raise ImageInputError(f"{image} exceeds the base64 request limit")
    return f"data:image/png;base64,{encoded}", width, height


def parse_tool_call(response):
    """Parse the answer the way the authors' code does, returning a normalized point.

    rules: the text between the last <tool_call> pair must be JSON with a 2-value point or a
    4-value box, whose center is used; values are divided by 1000 with no range check, so an
    off-screen point scores as wrong rather than invalid. This is deliberately more lenient
    than parsing.py, because matching the published method is this mode's purpose.
    why: non-finite values are the one departure. They crash or can't be logged here, and they
    score as incorrect either way.
    returns: (point or None, invalid reason or None).
    """
    if response is None:
        return None, "empty"
    try:
        action = json.loads(response.split("<tool_call>\n")[-1].split("\n</tool_call>")[-2])
        coordinates = action["arguments"]["coordinate"]
        if len(coordinates) == 2:
            point_x, point_y = coordinates
        elif len(coordinates) == 4:
            x1, y1, x2, y2 = coordinates
            point_x = (x1 + x2) / 2
            point_y = (y1 + y2) / 2
        else:
            raise ValueError("Wrong output format")
        point = [point_x / _COORDINATE_SCALE, point_y / _COORDINATE_SCALE]
    except (IndexError, KeyError, TypeError, ValueError, OverflowError):
        return None, "tool_call_parse_failed"
    if not all(math.isfinite(value) for value in point):
        return None, "non_finite"
    return point, None


class ReferenceQwenPredictor:
    """Sends the screenshot with the authors' system prompt, then reads a tool-call answer."""

    def __init__(self, client: FireworksClient):
        self._client = client

    def predict(self, image, instruction):
        """Predict one point.

        side effects: one model call.
        raises: ImageInputError for an image that cannot be sent.
        """
        started = time.perf_counter()
        image_uri, width, height = encode_reference_image(image)
        try:
            call = self._client.complete(image_uri, instruction, system=SYSTEM_PROMPT)
        except ApiFailure as failure:
            return PredictionResult(
                point_px=None, point_normalized=None, calls=[],
                latency_seconds=failure.latency_seconds or time.perf_counter() - started,
                fallback_used=False, invalid_reason=None, error=failure.to_dict(),
            )
        point, invalid_reason = parse_tool_call(call["content"])
        return PredictionResult(
            point_px=normalized_to_pixels(point, width, height) if point else None,
            point_normalized=point,
            calls=[call],
            latency_seconds=call["latency_seconds"],
            fallback_used=False,
            invalid_reason=invalid_reason,
            error=None,
        )
