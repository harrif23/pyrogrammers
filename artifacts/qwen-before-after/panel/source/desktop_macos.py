import sys
from PIL import Image
from .contracts import Point
from .geometry import Transform


def capture():
    if sys.platform != "darwin":
        raise RuntimeError("Desktop demo currently supports macOS only")
    import mss
    import pyautogui
    with mss.mss() as screen:
        if len(screen.monitors) != 2:
            raise RuntimeError("Use a single display for the initial demo")
        monitor = screen.monitors[1]
        shot = screen.grab(monitor)
        image = Image.frombytes("RGB", shot.size, shot.rgb)
    size = pyautogui.size()
    return image, (size.width, size.height)


def click(point: Point, screenshot_size: tuple[int, int], desktop_size: tuple[int, int]):
    import pyautogui
    current = pyautogui.size()
    if tuple(current) != desktop_size:
        raise RuntimeError("Display geometry changed; capture again")
    mapped = Transform(0, 0, *desktop_size, *screenshot_size).to_source(point)
    # PyAutoGUI's default corner fail-safe remains enabled.
    pyautogui.click(mapped.x, mapped.y)
