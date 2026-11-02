from unittest.mock import Mock
import pytest
from flground.model import FireworksModel


@pytest.mark.parametrize("shell_override", [False, True])
def test_dotenv_configuration(tmp_path, monkeypatch, shell_override):
    monkeypatch.chdir(tmp_path)
    for name in ("FIREWORKS_API_KEY", "FLGROUND_MODEL", "FLGROUND_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / ".env").write_text(
        'FIREWORKS_API_KEY="test-file-key"\n'
        'FLGROUND_MODEL=test-model\n'
        'FLGROUND_BASE_URL=https://example.test/v1\n'
    )
    if shell_override:
        monkeypatch.setenv("FIREWORKS_API_KEY", "test-shell-key")
    client = Mock()
    monkeypatch.setattr("openai.OpenAI", client)
    model = FireworksModel()
    assert model.model == "test-model"
    assert client.call_args.kwargs["api_key"] == (
        "test-shell-key" if shell_override else "test-file-key"
    )
    assert client.call_args.kwargs["base_url"] == "https://example.test/v1"


@pytest.mark.parametrize("raw", [
    '{"x": 3, "y": 4}',
    '```json\n[\n {"point_2d": [3, 4], "label": "snare channel volume control"}\n]\n```',
])
def test_deployment_request(tmp_path, monkeypatch, raw):
    from types import SimpleNamespace
    from PIL import Image
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setenv("FLGROUND_MODEL", "accounts/yuxinz721-ofi0hn0kj3/deployments/h0xbbw9t")
    monkeypatch.setenv("FLGROUND_MAX_TOKENS", "32768")
    constructor = Mock()
    monkeypatch.setattr("openai.OpenAI", constructor)
    create = constructor.return_value.chat.completions.create
    create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=raw))], usage=None)
    result = FireworksModel().predict(Image.new("RGB", (10, 10)), "volume")
    assert result.point.x == 3
    assert result.point.y == 4
    assert result.raw_response == raw
    create.assert_called_once()
    request = create.call_args.kwargs
    assert request["model"].endswith("/deployments/h0xbbw9t")
    assert request["max_tokens"] == 32768
    assert request["extra_body"] == {"top_k": 40}
    assert request["presence_penalty"] == request["frequency_penalty"] == 0
    assert "temperature" not in request
    assert request["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_transport_failure_retains_sanitized_trace(tmp_path, monkeypatch):
    from PIL import Image
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('FIREWORKS_API_KEY', 'test-key')
    monkeypatch.setenv('FLGROUND_MODEL', 'test-model')
    constructor = Mock()
    monkeypatch.setattr('openai.OpenAI', constructor)
    constructor.return_value.chat.completions.create.side_effect = RuntimeError('private request data')
    model = FireworksModel()
    with pytest.raises(RuntimeError):
        model.predict(Image.new('RGB',(10,10)), 'target')
    assert len(model.records) == 1
    record = model.records[0]
    assert record.error == 'RuntimeError'
    assert 'private request data' not in record.model_dump_json()
    assert len(record.image_hashes[0]) == 64


def test_normalized_response_and_truncation(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from PIL import Image
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('FIREWORKS_API_KEY','test-key')
    monkeypatch.setenv('FLGROUND_MODEL','test-model')
    monkeypatch.setenv('FLGROUND_COORDINATE_SPACE','normalized_1000')
    constructor = Mock()
    monkeypatch.setattr('openai.OpenAI', constructor)
    choice = SimpleNamespace(message=SimpleNamespace(content='{"x":250,"y":500}'),finish_reason='stop')
    constructor.return_value.chat.completions.create.return_value = SimpleNamespace(choices=[choice],usage=None)
    model = FireworksModel()
    result = model.predict(Image.new('RGB',(1600,800)), 'target')
    assert (result.point.x,result.point.y)==(400,400)
    assert '0 to 1000' in result.prompt
    choice.finish_reason='length'
    with pytest.raises(ValueError,match='Incomplete'):
        model.predict(Image.new('RGB',(1600,800)), 'target')
    assert model.records[-1].finish_reason=='length'
    assert model.records[-1].error=='ValueError'


def test_explicit_greedy_setting_is_sent_and_recorded(tmp_path, monkeypatch):
    from PIL import Image
    from types import SimpleNamespace
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('FIREWORKS_API_KEY','test-key')
    monkeypatch.setenv('FLGROUND_MODEL','test-model')
    monkeypatch.setenv('FLGROUND_TEMPERATURE','0')
    constructor=Mock()
    monkeypatch.setattr('openai.OpenAI',constructor)
    create=constructor.return_value.chat.completions.create
    create.return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"x":3,"y":4}'),finish_reason='stop')],usage=None)
    model=FireworksModel()
    model.predict(Image.new('RGB',(10,10)),'target')
    assert create.call_args.kwargs['temperature']==0
    assert model.manifest()['temperature']==0
