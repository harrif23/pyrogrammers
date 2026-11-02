"""Shortcut transitions kept independent of native event taps."""
class HoldShortcut:
    def __init__(self):
        self.held=False

    def event(self, kind, key=None, option=False, other_modifiers=False):
        if kind=='down' and key==6 and option and not other_modifiers:
            if self.held:return 'swallow'
            self.held=True
            return 'start'
        if self.held and ((kind=='up' and key==6) or (kind=='flags' and not option)):
            self.held=False
            return 'release'
        if kind=='down' and key==53:
            self.held=False
            return 'cancel'
        return None
