"""Run independently; no live server, microphone, display or benchmark files needed."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

HERE=Path(__file__).parent
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(HERE.parent/'src'))
from local_model import LocalModel
from backend import is_fl_studio


def fake_client(monkeypatch):
    client=Mock()
    client.models.list.return_value=SimpleNamespace(data=[SimpleNamespace(id='gemma-local')])
    client.chat.completions.create.return_value=SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"x":500,"y":250}'),finish_reason='stop')],usage=None)
    constructor=Mock(return_value=client)
    monkeypatch.setattr('local_model.OpenAI',constructor)
    return constructor,client


def test_local_route_ignores_fireworks_env(monkeypatch):
    monkeypatch.setenv('FLGROUND_MODEL','must-not-use-remote')
    monkeypatch.setenv('FIREWORKS_API_KEY','must-not-use-secret')
    constructor,client=fake_client(monkeypatch)
    model=LocalModel()
    result=model.predict(Image.new('RGB',(1000,800)),'Drums button')
    assert (result.point.x,result.point.y)==(500,200)
    assert constructor.call_args.kwargs['base_url']=='http://127.0.0.1:8080/v1'
    assert constructor.call_args.kwargs['api_key']=='local'
    request=client.chat.completions.create.call_args.kwargs
    assert request['model']=='gemma-local'
    assert request['messages'][0]['content'][1]['image_url']['url'].startswith('data:image/png;base64,')
    assert model.manifest()['provider']=='local llama-server'
    assert len(model.records)==1


@pytest.mark.parametrize('url',['https://example.com/v1','http://192.168.1.2:8080/v1',
                                'http://user:pass@localhost:8080/v1','file:///tmp/server'])
def test_no_silent_remote_routing(url):
    with pytest.raises(ValueError):LocalModel(url)


def test_multiple_models_need_explicit_configuration(monkeypatch):
    constructor,client=fake_client(monkeypatch)
    client.models.list.return_value=SimpleNamespace(data=[])
    with pytest.raises(ValueError):LocalModel()


def test_foreground_gate():
    assert is_fl_studio('FL Studio')
    assert is_fl_studio('FL Studio 2025')
    assert not is_fl_studio('Terminal')
