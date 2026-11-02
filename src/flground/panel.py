"""Experimental panel-first localization, independent of hosted provider logic."""
import math
import time
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict, model_validator

from .contracts import GroundingResult, Point
from .geometry import Transform
from .knowledge import VERSION, SOURCES, guidance
from .model import coordinate_instruction, parse_json, parse_point, to_pixels


class Box(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False, strict=True)
    left: float
    top: float
    right: float
    bottom: float

    def pixels(self, size, space):
        sx, sy = (1, 1) if space == 'pixels' else (
            size[0] / (1000 if space == 'normalized_1000' else 1),
            size[1] / (1000 if space == 'normalized_1000' else 1))
        bounds = (self.left*sx, self.top*sy, self.right*sx, self.bottom*sy)
        l, t, r, b = bounds
        if not (0 <= l < r <= size[0] and 0 <= t < b <= size[1]):
            raise ValueError('Invalid region geometry')
        return bounds


class PanelPlan(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    visible: bool
    panel: str
    entity: str
    control: str | None
    evidence: str
    region: Box | None
    anchor: Box | None
    point: Point | None

    @model_validator(mode='after')
    def coherent(self):
        if self.visible and not self.control:
            raise ValueError("Visible target requires a control")
        if self.visible and (self.region is None or self.point is None or not self.evidence.strip()):
            raise ValueError('Visible target requires region, point and evidence')
        if not self.visible and any(v is not None for v in (self.region, self.anchor, self.point)):
            raise ValueError('Absent target must not include geometry')
        return self


class Choice(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    choice: Literal['initial', 'refined', 'neither']
    evidence: str


def context_crop(region, anchor, size):
    """Retain both the predicted panel and its visible anchor, adding 5% context."""
    boxes = [region] + ([anchor] if anchor else [])
    l, t = min(b[0] for b in boxes), min(b[1] for b in boxes)
    r, b = max(b[2] for b in boxes), max(b[3] for b in boxes)
    pad = max(8, math.ceil(max(r-l, b-t)*0.05))
    # Only the crop context is clipped. Model coordinates/boxes are validated first.
    return (max(0, math.floor(l)-pad), max(0, math.floor(t)-pad),
            min(size[0], math.ceil(r)+pad), min(size[1], math.ceil(b)+pad))


def marked(image, point, label):
    view = image.copy()
    draw = ImageDraw.Draw(view)
    x, y = point.x, point.y
    draw.ellipse((x-10, y-10, x+10, y+10), outline='cyan', width=2)
    draw.text((max(0, x+12), max(0, y-18)), label, fill='cyan')
    return view


def panel_ground(image, instruction, model, strategy, artifacts: Path | None):
    start = time.perf_counter()
    space = getattr(model, 'coordinate_space', 'pixels')
    rules = '' if strategy == 'panel_plain' else guidance(instruction)
    prompt = (
        f'Locate the requested visible FL Studio control in this {image.width}x{image.height} screenshot. '
        f'{coordinate_instruction(space)}\nInstruction: {instruction}\n{rules}\n'
        'Identify the relevant panel and exact named entity before predicting the control. '
        'Choose a region that retains its label/header and adjacent controls, not a tight crop of the point. '
        'Use visible=false if the requested target cannot be identified; never substitute a different entity. '
        'Return only JSON with keys visible (boolean), panel, entity, control, evidence (short visible facts), '
        'region, anchor, point. region and anchor use {left,top,right,bottom}; point uses {x,y}. '
        'anchor is the visible entity label/header box, or null if unnecessary. '
        'For visible=false, region, anchor and point must all be null.')
    if strategy == 'panel_fast':
        prompt += (' Keep evidence to one short sentence. All point AND box coordinates must use '
                   'the coordinate units specified above, never mix pixels and normalized units.')
    first = model.complete([image], prompt, 'panel_plan')
    calls = [first]
    metadata = {'coordinate_space': space, 'knowledge_version': VERSION if rules else None,
                'knowledge_sources': SOURCES if rules else [], 'knowledge': rules}

    def result(point):
        return GroundingResult(instruction=instruction, image_width=image.width,
                               image_height=image.height, point=point, strategy=strategy,
                               model=model.model, calls=calls, status='ok' if point else 'not_found',
                               latency_seconds=time.perf_counter()-start, metadata=metadata)
    try:
        plan = PanelPlan.model_validate(parse_json(first.raw_response), strict=True)
        metadata['plan'] = plan.model_dump()
        if not plan.visible:
            return result(None)
        first.point = to_pixels(plan.point, image.size, space)
        region = plan.region.pixels(image.size, space)
        anchor = plan.anchor.pixels(image.size, space) if plan.anchor else None
        if not (region[0] <= first.point.x < region[2] and region[1] <= first.point.y < region[3]):
            raise ValueError('Initial point is outside selected panel')
    except Exception as exc:
        first.error = type(exc).__name__
        raise
    if strategy == 'panel_fast':
        metadata['localization'] = 'single_pass_panel_plan'
        return result(first.point)
    bounds = context_crop(region, anchor, image.size)
    crop = image.crop(bounds)
    # Bound enlargement; never shrink a native crop or imply that new detail is created.
    scale = min(2.0, max(1.0, 2048 / max(crop.size)))
    enlarged = crop.resize((round(crop.width*scale), round(crop.height*scale)), Image.Resampling.LANCZOS)
    transform = Transform(bounds[0], bounds[1], *crop.size, *enlarged.size)
    metadata['crop'] = {'bounds': bounds, 'output_size': enlarged.size, 'scale': scale}
    if artifacts:
        artifacts.mkdir(parents=True, exist_ok=True)
        enlarged.save(artifacts/'refinement.png')
    second = model.complete([enlarged], (
        f'Locate the requested control in this {enlarged.width}x{enlarged.height} crop. '
        f'{coordinate_instruction(space)} Coordinates refer ONLY to this crop. '
        f'Instruction: {instruction}\n{rules}\n'
        f'Proposed context (verify visually): panel={plan.panel}; entity={plan.entity}; control={plan.control}. '
        'Return only {"x":number,"y":number}, or {"visible":false} if the exact target is absent.'), 'panel_localize')
    calls.append(second)
    try:
        if parse_json(second.raw_response) == {'visible': False}:
            metadata['localization'] = 'target_not_visible'
            return result(None)
        second.point = to_pixels(parse_point(second.raw_response), enlarged.size, space)
        refined = transform.to_source(second.point)
    except Exception as exc:
        second.error = type(exc).__name__
        raise
    metadata['candidates'] = {'initial': first.point.model_dump(), 'refined': refined.model_dump()}
    if strategy != 'panel_verify':
        return result(refined)
    initial_local = Point(x=(first.point.x-bounds[0])*enlarged.width/crop.width,
                          y=(first.point.y-bounds[1])*enlarged.height/crop.height)
    # Include global context and separate marked local views; do not occlude targets with filled dots.
    views = [image, marked(enlarged, initial_local, 'initial'), marked(enlarged, second.point, 'refined')]
    if artifacts:
        views[1].save(artifacts/'candidate_initial.png')
        views[2].save(artifacts/'candidate_refined.png')
    verification = model.complete(views, (
        f'Instruction: {instruction}\n{rules}\n'
        'Image 1 is full-screen context. Images 2 and 3 are the same crop, marked initial and refined. '
        'Check the center of each ring against the exact requested panel, named entity and control function. '
        'Choose initial, refined, or neither; do not prefer refined merely because it is newer. '
        'Return only {"choice":"initial|refined|neither","evidence":"short visible facts"}.'), 'panel_verify')
    calls.append(verification)
    try:
        choice = Choice.model_validate(parse_json(verification.raw_response), strict=True)
    except Exception as exc:
        verification.error = type(exc).__name__
        raise
    metadata['verification'] = choice.model_dump()
    return result({'initial': first.point, 'refined': refined, 'neither': None}[choice.choice])
