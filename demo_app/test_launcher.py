from pathlib import Path
from unittest.mock import Mock
import subprocess
import sys

import pytest
sys.path.insert(0,str(Path(__file__).parent))
import launch


def test_unified_binary_uses_serve():
    command=launch.server_command(Path('/tmp/llama'),Path('/m.gguf'),Path('/p.gguf'),8080)
    assert command[:2]==['/tmp/llama','serve']
    assert command[command.index('-c')+1]=='65536'
    assert command[command.index('--host')+1]=='127.0.0.1'


def test_standalone_server_has_no_subcommand():
    assert launch.server_command(Path('/tmp/llama-server'),Path('/m'),Path('/p'),8080)[1]=='-m'


def test_readiness_rejects_other_model(monkeypatch):
    monkeypatch.setattr(launch,'get_json',lambda url: {'status':'ok'} if url.endswith('/health') else {'data':[{'id':'other.gguf'}]})
    with pytest.raises(RuntimeError,match='different'):
        launch.ready('http://localhost:8080',Path('gemma.gguf'))


def test_readiness_accepts_full_model_path(monkeypatch):
    monkeypatch.setattr(launch,'get_json',lambda url: {'status':'ok'} if url.endswith('/health') else {'data':[{'id':'/models/gemma.gguf'}]})
    assert launch.ready('http://localhost:8080',Path('gemma.gguf'))


def test_cleanup_only_owned_handle():
    launch.stop(None)
    process=Mock()
    process.poll.return_value=None
    process.wait.side_effect=[subprocess.TimeoutExpired('server',5),0]
    launch.stop(process)
    process.terminate.assert_called_once()
    process.kill.assert_called_once()
