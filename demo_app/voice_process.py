"""Launch speech as a macOS application so TCC attributes permissions to its bundle."""
from pathlib import Path
import subprocess
import time
import json


class StopSignal:
    def __init__(self, path): self.path = path
    def write(self, value): self.path.touch()
    def flush(self): pass


class VoiceProcess:
    def __init__(self, executable, logs, endpoint="http://127.0.0.1:8080/v1"):
        self.endpoint = endpoint
        logs.mkdir(parents=True, exist_ok=True)
        session = logs/f'voice-{time.time_ns()}'
        self.output = session.with_suffix('.jsonl')
        self.log = session.with_suffix('.log')
        self.stop_file = session.with_suffix('.stop')
        self.output.touch()
        self.log.touch()
        self.stdin = StopSignal(self.stop_file)
        bundle = Path(executable).parents[2]
        self.process = subprocess.Popen(['/usr/bin/open','-n','-W',str(bundle),
            '--args','--events-file',str(self.output),'--log-file',str(self.log),
            '--stop-file',str(self.stop_file),'--audio-file',str(session.with_suffix('.wav'))])
        self.stdout = self.lines()
    def lines(self):
        with self.output.open() as stream:
            while True:
                line = stream.readline()
                if line:
                    event=json.loads(line)
                    if event.get('event')=='audio':
                        try:
                            from transcribe import transcribe
                            text=transcribe(event['path'],self.endpoint)
                            yield json.dumps({'event':'transcript','text':text})+'\n'
                        except Exception as exc:
                            with self.log.open('a') as log:
                                log.write(f'Gemma transcription failed: {type(exc).__name__}\n')
                            yield json.dumps({'event':'error','message':f'Gemma transcription failed ({type(exc).__name__}). See voice log.'})+'\n'
                    else:
                        yield line
                elif self.poll() is not None:
                    break
                else:
                    time.sleep(.03)
    def poll(self): return self.process.poll()
    def wait(self): return self.process.wait()
    def terminate(self): self.stop_file.touch()
