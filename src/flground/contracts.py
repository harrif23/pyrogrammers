from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class Point(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    x: float
    y: float


class CallRecord(BaseModel):
    point: Point
    raw_response: str
    latency_seconds: float
    usage: dict = Field(default_factory=dict)


class GroundingResult(BaseModel):
    instruction: str
    image_width: int
    image_height: int
    point: Point
    strategy: Literal["baseline", "refine"]
    model: str
    calls: list[CallRecord]
    latency_seconds: float

    @property
    def model_calls(self) -> int:
        return len(self.calls)
