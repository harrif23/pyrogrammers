#!/usr/bin/env python3
"""Start local Gemma, wait for readiness, and run the native FLGround app."""
import argparse
import json
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = Path.home()/'.cache/huggingface/hub/models--ggml-org--gemma-4-E2B-it-GGUF/snapshots/a1dac71d3ab220618f5a7573a52acdc4baf3ae3b'


def server_command(binary, model, projector, port):
    command = [str(binary)]
    if binary.name == 'llama':
        command.append('serve')
    return command + ['-m',str(model),'--mmproj',str(projector),'-ngl','99','-c','65536',
                      '--host','127.0.0.1','--port',str(port),'-b','4096','-ub','2048']


def get_json(url):
    # Avoid proxy environment variables for the local server.
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url,timeout=2) as response:
        return json.load(response)


def ready(base, model):
    if get_json(base+'/health').get('status') != 'ok':
        return False
    models=get_json(base+'/v1/models').get('data',[])
    if len(models)!=1 or Path(models[0]['id']).name not in (model.name,model.stem):
        raise RuntimeError('Port is serving a different/aliased model. Stop it yourself or choose another --port.')
    return True


def stop(process):
    if process is not None and process.poll() is None:
        process.terminate()
        try:process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill();process.wait()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=Path,default=Path.home()/'.local/bin/llama')
    parser.add_argument('--model',type=Path,default=SNAPSHOT/'gemma-4-E2B-it-Q8_0.gguf')
    parser.add_argument('--mmproj',type=Path,default=SNAPSHOT/'mmproj-gemma-4-E2B-it-Q8_0.gguf')
    parser.add_argument('--port',type=int,default=8080)
    parser.add_argument('--timeout',type=int,default=180)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    if not 1<=args.port<=65535 or args.timeout<=0:
        parser.error('Port must be 1–65535 and timeout must be positive')
    python=ROOT/'.venv/bin/python'
    for path in (args.binary,args.model,args.mmproj,python):
        if not path.is_file():parser.error(f'Missing file: {path}')
    command=server_command(args.binary,args.model,args.mmproj,args.port)
    base=f'http://127.0.0.1:{args.port}'
    app_command=[str(python),str(ROOT/'demo_app/hotkey_app.py'),'--endpoint',base+'/v1']
    if args.dry_run:
        print(json.dumps({'server':command,'app':app_command,'readiness':base+'/health'},indent=2))
        print('Dry run: nothing launched.');return 0
    server=app=None
    log=None
    def interrupted(signum,frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted)
    try:
        helper=ROOT/'artifacts/demo-build/FLGround Voice.app/Contents/MacOS/FLGroundVoice'
        sources=[ROOT/'demo_app/Speech.swift',ROOT/'demo_app/build.py']
        if not helper.exists() or any(p.stat().st_mtime>helper.stat().st_mtime for p in sources):
            if shutil.which('swiftc'):
                try:subprocess.run([str(python),str(ROOT/'demo_app/build.py')],check=True,cwd=ROOT)
                except subprocess.CalledProcessError:
                    print('Voice build failed; typed input remains available.',flush=True)
            else:print('Swift compiler not found; typed input remains available.',flush=True)
        with socket.socket() as sock:
            sock.settimeout(1)
            occupied=sock.connect_ex(('127.0.0.1',args.port))==0
        if occupied:
            print(f'Checking existing server on port {args.port}; it will not be stopped on exit.',flush=True)
        else:
            logs=ROOT/'artifacts/demo-build'
            logs.mkdir(parents=True,exist_ok=True)
            log_path=logs/f'llama-{time.time_ns()}.log'
            log=log_path.open('w')
            server=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
            print(f'Starting Gemma. Server log: {log_path}',flush=True)
        deadline=time.monotonic()+args.timeout
        while True:
            if server is not None and server.poll() is not None:
                raise RuntimeError('Gemma exited during startup; inspect the server log above.')
            try:
                if ready(base,args.model):break
            except (OSError,ValueError):pass
            if time.monotonic()>=deadline:
                raise RuntimeError(f'Server not ready after {args.timeout}s. Check the log or occupied port.')
            time.sleep(.5)
        print('Gemma is ready. Hold Option+Z in FL Studio to speak. Quit from the FL menu or press Ctrl-C here to finish.',flush=True)
        app=subprocess.Popen(app_command,cwd=ROOT)
        while app.poll() is None:
            if server is not None and server.poll() is not None:
                raise RuntimeError('Gemma stopped unexpectedly; closing the frontend.')
            time.sleep(.5)
        return app.returncode
    except KeyboardInterrupt:
        print('\nClosing demo…',flush=True);return 130
    except RuntimeError as exc:
        print(str(exc),flush=True);return 1
    finally:
        stop(app)
        stop(server)
        if log:log.close()


if __name__=='__main__':
    raise SystemExit(main())
