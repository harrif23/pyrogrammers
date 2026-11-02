import time
from pathlib import Path
from typing import Literal
from PIL import Image
from .contracts import GroundingResult
from .geometry import Transform, crop_bounds, validate_point
from .model import GroundingModel


def ground(image: Image.Image, instruction: str, model: GroundingModel,
           strategy: Literal["baseline", "refine"] = "baseline", crop_size: int = 512,
           artifacts: Path | None = None) -> GroundingResult:
    if strategy not in ("baseline", "refine"):
        raise ValueError(f"Unknown strategy: {strategy}")
    if not instruction.strip():
        raise ValueError("Instruction must not be empty")
    start = time.perf_counter()
    first = model.predict(image, instruction)
    validate_point(first.point, *image.size)
    calls, point = [first], first.point
    if strategy == "refine":
        bounds = crop_bounds(point, *image.size, crop_size)
        crop = image.crop(bounds)
        enlarged = crop.resize((crop.width * 2, crop.height * 2), Image.Resampling.LANCZOS)
        if artifacts:
            artifacts.mkdir(parents=True, exist_ok=True)
            enlarged.save(artifacts / "refinement.png")
        second = model.predict(enlarged, instruction)
        calls.append(second)
        point = Transform(bounds[0], bounds[1], crop.width, crop.height,
                          enlarged.width, enlarged.height).to_source(second.point)
    validate_point(point, *image.size)
    return GroundingResult(instruction=instruction, image_width=image.width,
                           image_height=image.height, point=point, strategy=strategy,
                           model=model.model, calls=calls,
                           latency_seconds=time.perf_counter() - start)
