"""Small, versioned manual-derived rules, with no benchmark examples or coordinates."""

VERSION = 'fl-manual-v1'
SOURCES = [
    'https://www.image-line.com/fl-studio-learning/fl-studio-online-manual/html/channelrack.htm',
    'https://www.image-line.com/fl-studio-learning/fl-studio-online-manual/html/mixer.htm',
]


def guidance(instruction: str) -> str:
    text = instruction.lower()
    rules = [
        'Use only visible evidence. Preserve the exact named channel, track or plugin; '
        'do not substitute Drums for Snare or assume a hidden target is visible.',
        'Distinguish Channel Rack, Mixer, Playlist, Piano roll, Browser, toolbar, '
        'plugin windows, dialogs and menus. Match the requested function and entity.',
    ]
    if any(word in text for word in ('channel', 'volume', 'pan', 'snare', 'drum')):
        rules.append('Channel Rack rows have separate enable, pan, volume, routing and '
                     'channel-name controls. The channel-name button opens its settings/plugin. '
                     'Channel volume is not the enable indicator, pan knob, or routing number.')
    if any(word in text for word in ('mixer', 'track', 'volume', 'master', 'fader')):
        rules.append('Mixer level faders control mixer tracks; match the visible track header. '
                     'Channel Rack volume, Mixer track level, and main monitor volume are '
                     'different controls. Playlist tracks do not establish Mixer routing.')
    return '\n'.join(rules)
