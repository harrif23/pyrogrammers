"""Standalone process: reuse the harness without modifying it or its environment."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def emit(**data):
    print(json.dumps(data), flush=True)


def front_app():
    from AppKit import NSWorkspace
    app = NSWorkspace.sharedWorkspace().frontmostApplication()
    return app.localizedName() if app else ''


def is_fl_studio(name):
    return name.lower().startswith('fl studio')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--instruction',required=True)
    parser.add_argument('--strategy',choices=['baseline','panel','panel_fast','panel_recover','panel_verify'],default='panel_recover')
    parser.add_argument('--delay',type=int,default=5)
    parser.add_argument('--endpoint',default='http://127.0.0.1:8080/v1')
    parser.add_argument('--coordinates',default='normalized_1000',choices=['pixels','normalized_1000','normalized_1'])
    args=parser.parse_args()
    if not args.instruction.strip() or len(args.instruction)>500:
        emit(event='error',message='Enter a request between 1 and 500 characters.');return
    os.chdir(ROOT)
    # Process-local settings only. Never edit .env or change the benchmark process.
    os.environ['FLGROUND_COORDINATE_SPACE']='normalized_1000'
    destination=ROOT/'artifacts/voice-demo'/uuid4().hex
    model=None
    try:
        from flground.desktop_macos import capture
        from flground.harness import ground
        from flground.model import FireworksModel
        from flground.reports import save_report
        os.environ['FLGROUND_MAX_TOKENS']='8192'
        os.environ['FLGROUND_TEMPERATURE']='0'
        model=FireworksModel()
        emit(event='model',name=model.model)
        for remaining in range(args.delay,0,-1):
            emit(event='countdown',seconds=remaining)
            time.sleep(1)
        if not is_fl_studio(front_app()):
            emit(event='error',message='Bring FL Studio to the front, then try again.');return
        image,desktop=capture()
        destination.mkdir(parents=True)
        image.save(destination/'capture.png')
        emit(event='working',message='Finding your control…')
        result=ground(image,args.instruction,model,args.strategy,artifacts=destination)
        from verify_target import verify_target
        result=verify_target(image,args.instruction,model,result,destination)
        save_report(image,result,destination)
        emit(event='result',point=result.point.model_dump() if result.point else None,
             screenshot_size=list(image.size),desktop_size=list(desktop),
             seconds=round(result.latency_seconds,1),report=str(destination/'report.html'),
             image=str(destination/'prediction.png'),calls=result.model_calls)
    except Exception as exc:
        destination.mkdir(parents=True,exist_ok=True)
        (destination/'failure.json').write_text(json.dumps({
            'error':type(exc).__name__,
            'calls':[c.model_dump() for c in model.records] if model else []},indent=2))
        if isinstance(exc, ValueError):
            message=f'Model returned an invalid target ({type(exc).__name__}). Try again.'
        else:
            message=f'Could not finish ({type(exc).__name__}). See the saved failure report.'
        print(f'{type(exc).__name__}: failure report {destination / "failure.json"}',file=sys.stderr,flush=True)
        emit(event='error',message=message,
             report=str(destination/'failure.json'))


if __name__=='__main__':
    main()
