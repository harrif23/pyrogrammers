from pathlib import Path
from unittest.mock import Mock
import sys
sys.path.insert(0,str(Path(__file__).parent))
from voice_process import VoiceProcess


def test_application_launch_and_release_signal(tmp_path, monkeypatch):
    process=Mock()
    process.poll.return_value=0
    process.wait.return_value=0
    launch=Mock(return_value=process)
    monkeypatch.setattr('voice_process.subprocess.Popen',launch)
    executable=tmp_path/'Voice.app/Contents/MacOS/Voice'
    voice=VoiceProcess(executable,tmp_path/'logs')
    command=launch.call_args.args[0]
    assert command[:4]==['/usr/bin/open','-n','-W',str(tmp_path/'Voice.app')]
    assert '--events-file' in command and '--stop-file' in command
    voice.output.write_text('{"event":"transcript","text":"Drums"}\n')
    assert list(voice.stdout)==['{"event":"transcript","text":"Drums"}\n']
    voice.stdin.write('\n'); voice.stdin.flush()
    assert voice.stop_file.exists()
    assert voice.wait()==0
