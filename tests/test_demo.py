from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image
from typer.testing import CliRunner

from flground.cli import app
from flground.contracts import Point
from flground.overlay_macos import marker_frame


@pytest.mark.parametrize('point, expected', [
    (Point(x=1000, y=600), ((460, 560), (80, 80))),
    (Point(x=0, y=0), ((-40, 860), (80, 80))),
    (Point(x=2878, y=1798), ((1399, -39), (80, 80))),
])
def test_retina_overlay_frame(point, expected):
    assert marker_frame(point, (2880, 1800), (1440, 900)) == expected


def test_overlay_rejects_invalid_point():
    with pytest.raises(ValueError):
        marker_frame(Point(x=2880, y=0), (2880, 1800), (1440, 900))


@pytest.mark.parametrize('click_requested, changed', [(False, False), (True, False), (True, True)])
def test_demo_overlay_and_click_safety(monkeypatch, click_requested, changed):
    events = []
    original = Image.new('RGB', (100, 80))
    fresh = Image.new('RGB', (100, 80), 'white') if changed else original.copy()
    frames = iter([original, fresh])

    def capture():
        events.append('capture')
        return next(frames), (50, 40)

    def predict(*args):
        events.append('predict')
        return SimpleNamespace(point=Point(x=30, y=20))

    def overlay(*args):
        events.append('overlay')
        assert args == (Point(x=30, y=20), (100, 80), (50, 40), 10)

    monkeypatch.setattr('flground.desktop_macos.capture', capture)
    click = Mock()
    monkeypatch.setattr('flground.desktop_macos.click', click)
    monkeypatch.setattr('flground.cli.run', predict)
    monkeypatch.setattr('flground.cli.time.sleep', lambda delay: events.append(('delay', delay)))
    monkeypatch.setattr('flground.overlay_macos.show_overlay', overlay)
    args = ['demo', '--instruction', 'snare volume']
    if click_requested:
        args.append('--execute-click')
    result = CliRunner().invoke(app, args, input='y\n')
    assert events[:4] == [('delay', 5), 'capture', 'predict', 'overlay']
    if click_requested:
        assert events[4:] == ['capture']
    else:
        assert len(events) == 4
    if click_requested and changed:
        assert result.exit_code != 0
        assert 'Screen changed' in result.output
    else:
        assert result.exit_code == 0, result.output
    if click_requested and not changed:
        click.assert_called_once_with(Point(x=30, y=20), (100, 80), (50, 40))
    else:
        click.assert_not_called()
