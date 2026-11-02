from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class Point(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    x: float
    y: float


class CallRecord(BaseModel):
    point: Point | None = None
    raw_response: str
    latency_seconds: float
    usage: dict = Field(default_factory=dict)
    stage: str = "predict"
    prompt: str = ""
    image_sizes: list[tuple[int, int]] = Field(default_factory=list)
    image_hashes: list[str] = Field(default_factory=list)
    finish_reason: str | None = None
    error: str | None = None


class GroundingResult(BaseModel):
    instruction: str
    image_width: int
    image_height: int
    point: Point | None
    strategy: Literal["baseline", "refine", "knowledge", "panel", "panel_verify", "panel_plain", "panel_fast", "panel_recover", "panel_compact", "panel_selective"]
    model: str
    calls: list[CallRecord]
    latency_seconds: float
    status: Literal["ok", "not_found"] = "ok"
    metadata: dict = Field(default_factory=dict)

    @property
    def model_calls(self) -> int:
        return len(self.calls)
