import base64
import io
import hashlib
import json
import os
import re
import time
from pathlib import Path
from dotenv import load_dotenv
from typing import Protocol
from PIL import Image
from .contracts import CallRecord, Point
from .geometry import validate_point


class GroundingModel(Protocol):
    model: str
    def predict(self, image: Image.Image, instruction: str) -> CallRecord: ...
    def complete(self, images: list[Image.Image], prompt: str, stage: str) -> CallRecord: ...


def parse_json(raw: str):
    text = raw.strip()
    fence = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    return json.loads(text)


def parse_point(raw: str) -> Point:
    """Parse one target; coordinate units are handled explicitly by the adapter."""
    data = parse_json(raw)
    if isinstance(data, list):
        if len(data) != 1:
            raise ValueError("Expected exactly one target in model response")
        data = data[0]
    if isinstance(data, dict) and "point_2d" in data:
        if set(data) - {"point_2d", "label"}:
            raise ValueError("Unexpected fields alongside point_2d")
        coordinates = data["point_2d"]
        if not isinstance(coordinates, list) or len(coordinates) != 2:
            raise ValueError("point_2d must contain exactly two pixel coordinates")
        data = {"x": coordinates[0], "y": coordinates[1]}
    return Point.model_validate(data, strict=True)


class FireworksModel:
    def __init__(self):
        from openai import OpenAI
        load_dotenv(Path.cwd() / ".env", override=False)
        self.model = os.environ.get("FLGROUND_MODEL", "")
        key = os.environ.get("FIREWORKS_API_KEY", "")
        if not self.model or not key:
            raise ValueError("Set FLGROUND_MODEL and FIREWORKS_API_KEY before inference")
        self.coordinate_space = os.environ.get("FLGROUND_COORDINATE_SPACE", "pixels")
        if self.coordinate_space not in ("pixels", "normalized_1000", "normalized_1"):
            raise ValueError("FLGROUND_COORDINATE_SPACE must be pixels, normalized_1000 or normalized_1")
        self.records: list[CallRecord] = []
        self.max_tokens = int(os.environ.get("FLGROUND_MAX_TOKENS", "32768"))
        if self.max_tokens <= 0:
            raise ValueError("FLGROUND_MAX_TOKENS must be positive")
        self.client = OpenAI(api_key=key, base_url=os.environ.get(
            "FLGROUND_BASE_URL", "https://api.fireworks.ai/inference/v1"),
            timeout=180, max_retries=0)

    def manifest(self) -> dict:
        return {"model": self.model, "coordinate_space": self.coordinate_space,
                "max_tokens": self.max_tokens, "top_k": 40, "presence_penalty": 0,
                "frequency_penalty": 0, "temperature": "endpoint_default",
                "timeout_seconds": 180, "max_retries": 0,
                "preprocessing": "RGB PNG; no client resize for baseline",
                "server_preprocessing": "not verified"}

    def complete(self, images: list[Image.Image], prompt: str, stage: str) -> CallRecord:
        content = [{"type": "text", "text": prompt}]
        hashes = []
        for image in images:
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, format="PNG")
            data = buffer.getvalue()
            hashes.append(hashlib.sha256(data).hexdigest())
            encoded = base64.b64encode(data).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}})
        record = CallRecord(raw_response="", latency_seconds=0, stage=stage, prompt=prompt,
                            image_sizes=[im.size for im in images], image_hashes=hashes)
        self.records.append(record)
        start = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=self.model, max_tokens=self.max_tokens,
                presence_penalty=0, frequency_penalty=0, extra_body={"top_k": 40},
                messages=[{"role": "user", "content": content}])
            record.usage = response.usage.model_dump() if response.usage else {}
            choice = response.choices[0]
            record.raw_response = choice.message.content or ""
            record.finish_reason = getattr(choice, "finish_reason", None)
            if record.finish_reason in ("length", "content_filter"):
                raise ValueError("Incomplete model response")
            return record
        except Exception as exc:
            record.error = type(exc).__name__
            raise
        finally:
            record.latency_seconds = time.perf_counter() - start

    def predict(self, image: Image.Image, instruction: str) -> CallRecord:
        prompt = (f"Locate the UI control described below in this {image.width}x{image.height} screenshot. "
                  "Return only a JSON object with numeric x and y pixel coordinates, "
                  "measured from the top-left of this exact image. Do not use normalized coordinates. "
                  f"Instruction: {instruction}")
        if self.coordinate_space != "pixels":
            prompt = (f"Locate the UI control in this {image.width}x{image.height} screenshot. "
                      f"Return only a JSON object with numeric x and y. {coordinate_instruction(self.coordinate_space)} "
                      f"Instruction: {instruction}")
        record = self.complete([image], prompt, "predict")
        try:
            record.point = to_pixels(parse_point(record.raw_response), image.size, self.coordinate_space)
        except Exception as exc:
            record.error = type(exc).__name__
            raise
        return record


def coordinate_instruction(space: str) -> str:
    if space == "pixels":
        return "Use original image pixel coordinates measured from the top-left; do not normalize."
    scale = 1000 if space == "normalized_1000" else 1
    return f"Use coordinates from 0 to {scale} relative to the image width/height, measured from the top-left."


def to_pixels(point: Point, size: tuple[int, int], space: str) -> Point:
    if space == "pixels":
        return validate_point(point, *size)
    if space not in ("normalized_1000", "normalized_1"):
        raise ValueError("Unknown coordinate space")
    scale = 1000 if space == "normalized_1000" else 1
    if not (0 <= point.x < scale and 0 <= point.y < scale):
        raise ValueError("Normalized point outside image")
    return validate_point(Point(x=point.x * size[0] / scale, y=point.y * size[1] / scale), *size)
