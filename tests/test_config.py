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
