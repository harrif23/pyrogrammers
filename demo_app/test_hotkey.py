from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent))
from hotkey_state import HoldShortcut


def test_hold_release_and_repeat():
    state=HoldShortcut()
    assert state.event('down',6,True)=='start'
    assert state.event('down',6,True)=='swallow'
    assert state.event('up',6,True)=='release'
    assert state.event('up',6,True) is None


def test_option_released_first_stops_recording():
    state=HoldShortcut()
    state.event('down',6,True)
    assert state.event('flags',option=False)=='release'
    assert state.event('up',6) is None


def test_unrelated_shortcuts_pass_through():
    state=HoldShortcut()
    assert state.event('down',6,False) is None
    assert state.event('down',6,True,True) is None
    assert state.event('down',7,True) is None


def test_escape_clears_hold():
    state=HoldShortcut()
    state.event('down',6,True)
    assert state.event('down',53)=='cancel'
    assert not state.held
    assert state.event('up',6) is None
