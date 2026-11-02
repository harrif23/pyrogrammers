"""Bounded ablations; no labels, saved baseline predictions or task-ID routing."""
import time

from PIL import Image
from pydantic import BaseModel, ConfigDict, model_validator

from .contracts import GroundingResult, Point
from .geometry import crop_bounds
from .knowledge import guidance
from .model import coordinate_instruction, parse_json, parse_point, to_pixels


class CompactPlan(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    visible: bool
    panel: str
    entity: str
    point: Point | None
    ambiguous: bool

    @model_validator(mode='after')
    def coherent(self):
        if self.visible != (self.point is not None):
            raise ValueError('Visibility and point disagree')
        return self


def experimental_ground(image, instruction, model, strategy, artifacts=None):
    start = time.perf_counter()
    space = getattr(model, 'coordinate_space', 'pixels')
    calls = []
    metadata = {}

    def result(point):
        return GroundingResult(instruction=instruction, image_width=image.width,
                               image_height=image.height, point=point, strategy=strategy,
                               model=model.model, calls=calls, metadata=metadata,
                               status='ok' if point else 'not_found',
                               latency_seconds=time.perf_counter()-start)

    if strategy == 'panel_recover':
        from .panel import panel_ground
        offset = len(getattr(model, 'records', []))
        try:
            original = panel_ground(image, instruction, model, 'panel_fast', artifacts)
        except ValueError:
            failed = getattr(model, 'records', [])[offset:]
            # Only retry explicit token exhaustion, never API errors or bad geometry.
            if len(failed) != 1 or failed[0].finish_reason != 'length':
                raise
            calls.extend(failed)
            metadata['recovery_reason'] = 'length'
            recovery = model.predict(image, instruction)
            calls.append(recovery)
            return result(recovery.point)
        original.strategy = strategy
        return original

    prompt = (
        f'Locate this FL Studio control: {instruction}\n{guidance(instruction)}\n'
        f'{coordinate_instruction(space)} '
        'Identify the panel and exact named entity; distinguish neighboring controls by function. '
        'Return only {"visible":boolean,"panel":"short name","entity":"short name",'
        '"point":{"x":number,"y":number},"ambiguous":boolean}. '
        'Set point=null if absent. Set ambiguous=true if multiple visible controls plausibly '
        'match the request. No explanation or bounding boxes.')
    first = model.complete([image], prompt, 'compact_plan')
    calls.append(first)
    try:
        plan = CompactPlan.model_validate(parse_json(first.raw_response), strict=True)
        point = to_pixels(plan.point, image.size, space) if plan.visible else None
        first.point = point
    except Exception as exc:
        first.error = type(exc).__name__
        raise
    metadata['plan'] = plan.model_dump()
    if strategy == 'panel_compact' or point is None or not plan.ambiguous:
        metadata['verification'] = 'not_requested'
        return result(point)

    bounds = crop_bounds(point, *image.size, 768)
    crop = image.crop(bounds)
    enlarged = crop.resize((crop.width*2, crop.height*2), Image.Resampling.LANCZOS)
    metadata['verification_crop'] = {'bounds': bounds, 'output_size': enlarged.size}
    if artifacts:
        artifacts.mkdir(parents=True, exist_ok=True)
        enlarged.save(artifacts/'verification.png')
    second = model.complete([image, enlarged], (
        f'Request: {instruction}. Image 1 is the full screenshot. Image 2 is an enlarged '
        f'detail from original pixel bounds {bounds}. Check the exact named entity and '
        'control function against neighboring controls; do not assume the first guess is correct. '
        f'Return coordinates relative ONLY to IMAGE 1. {coordinate_instruction(space)} '
        'Return only {"x":number,"y":number}, or {"visible":false} if unidentified.'),
        'compact_verify')
    calls.append(second)
    try:
        refined = (None if parse_json(second.raw_response) == {'visible': False} else
                   to_pixels(parse_point(second.raw_response), image.size, space))
        second.point = refined
    except Exception as exc:
        second.error = type(exc).__name__
        raise
    metadata['verification'] = 'model_reported_ambiguity'
    metadata['initial_point'] = point.model_dump()
    return result(refined)
