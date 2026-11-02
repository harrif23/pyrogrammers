"""Demo-only wider-context localization; not part of reported benchmark scores."""
import time
from flground.geometry import crop_bounds, Transform
from flground.model import coordinate_instruction, parse_json, parse_point, to_pixels


def verify_target(image, instruction, model, result, destination):
    if result.point is None:
        return result
    start=time.perf_counter()
    bounds=crop_bounds(result.point,*image.size,768)
    crop=image.crop(bounds)
    crop.save(destination/'verification-crop.png')
    call=model.complete([crop],
        f'Locate the requested control in this screenshot crop: {instruction}. '
        'Check its visible label and function against neighboring controls. '
        'For tempo, identify the numeric BPM field, not the time display or transport buttons. '
        f'{coordinate_instruction(model.coordinate_space)} Coordinates refer ONLY to this crop. '
        'Return only {"x":number,"y":number}, or {"visible":false} if the target is not visible.',
        'demo_context_verify')
    result.calls.append(call)
    try:
        if parse_json(call.raw_response)=={'visible':False}:
            point=None
        else:
            call.point=to_pixels(parse_point(call.raw_response),crop.size,model.coordinate_space)
            point=Transform(bounds[0],bounds[1],*crop.size,*crop.size).to_source(call.point)
    except Exception as exc:
        call.error=type(exc).__name__
        raise
    result.metadata['demo_context_verification']={'bounds':bounds,'initial_point':result.point.model_dump()}
    result.point=point
    result.status='ok' if point else 'not_found'
    result.latency_seconds+=time.perf_counter()-start
    return result
