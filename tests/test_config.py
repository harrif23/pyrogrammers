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


def test_deployment_request(tmp_path, monkeypatch):
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
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"x": 3, "y": 4}'))], usage=None)
    result = FireworksModel().predict(Image.new("RGB", (10, 10)), "volume")
    assert result.point.x == 3
    request = create.call_args.kwargs
    assert request["model"].endswith("/deployments/h0xbbw9t")
    assert request["max_tokens"] == 32768
    assert request["extra_body"] == {"top_k": 40}
    assert request["presence_penalty"] == request["frequency_penalty"] == 0
    assert "temperature" not in request
    assert request["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
