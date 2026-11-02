import time
from pathlib import Path
from typing import Literal
from PIL import Image
from .contracts import GroundingResult
from .geometry import Transform, crop_bounds, validate_point
from .model import GroundingModel


def ground(image: Image.Image, instruction: str, model: GroundingModel,
           strategy: str = "baseline", crop_size: int = 512,
           artifacts: Path | None = None) -> GroundingResult:
    if strategy not in ("baseline", "refine", "knowledge", "panel", "panel_plain", "panel_fast", "panel_recover", "panel_compact", "panel_selective", "panel_verify"):
        raise ValueError(f"Unknown strategy: {strategy}")
    if not instruction.strip():
        raise ValueError("Instruction must not be empty")
    if strategy in ('panel_recover', 'panel_compact', 'panel_selective'):
        from .experimental import experimental_ground
        return experimental_ground(image, instruction, model, strategy, artifacts)
    if strategy.startswith("panel"):
        from .panel import panel_ground
        return panel_ground(image, instruction, model, strategy, artifacts)
    start = time.perf_counter()
    prompt_instruction = instruction
    if strategy == "knowledge":
        from .knowledge import guidance
        prompt_instruction += "\nFL Studio context:\n" + guidance(instruction)
    first = model.predict(image, prompt_instruction)
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
