import base64
import io
import os
import time
from typing import Protocol
from PIL import Image
from .contracts import CallRecord, Point
from .geometry import validate_point


class GroundingModel(Protocol):
    model: str
    def predict(self, image: Image.Image, instruction: str) -> CallRecord: ...


class FireworksModel:
    def __init__(self):
        from openai import OpenAI
        self.model = os.environ.get("FLGROUND_MODEL", "")
        key = os.environ.get("FIREWORKS_API_KEY", "")
        if not self.model or not key:
            raise ValueError("Set FLGROUND_MODEL and FIREWORKS_API_KEY before inference")
        self.client = OpenAI(api_key=key, base_url=os.environ.get(
            "FLGROUND_BASE_URL", "https://api.fireworks.ai/inference/v1"),
            timeout=60, max_retries=0)

    def predict(self, image: Image.Image, instruction: str) -> CallRecord:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        prompt = (f"Locate the UI control described below in this {image.width}x{image.height} screenshot. "
                  "Return only a JSON object with numeric x and y pixel coordinates, "
                  "measured from the top-left of this exact image. Do not use normalized coordinates. "
                  f"Instruction: {instruction}")
        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model, temperature=0, max_tokens=256,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}]}])
        raw = response.choices[0].message.content or ""
        point = Point.model_validate_json(raw)
        validate_point(point, image.width, image.height)
        return CallRecord(point=point, raw_response=raw,
                          latency_seconds=time.perf_counter() - start,
                          usage=response.usage.model_dump() if response.usage else {})
