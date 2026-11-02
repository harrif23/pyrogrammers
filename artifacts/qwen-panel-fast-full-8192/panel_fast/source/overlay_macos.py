"""Temporary, non-activating macOS target marker; never moves or clicks the mouse."""

import io
import sys
import time

from PIL import Image, ImageDraw

from .contracts import Point
from .geometry import Transform


def marker_frame(point: Point, screenshot_size: tuple[int, int],
                 desktop_size: tuple[int, int], diameter: int = 80):
    """Map top-left screenshot pixels to a bottom-left Cocoa window rectangle."""
    mapped = Transform(0, 0, *desktop_size, *screenshot_size).to_source(point)
    return ((mapped.x - diameter / 2, desktop_size[1] - mapped.y - diameter / 2),
            (diameter, diameter))


def show_overlay(point: Point, screenshot_size: tuple[int, int],
                 desktop_size: tuple[int, int], seconds: int = 10):
    if sys.platform != "darwin":
        raise RuntimeError("On-screen overlays currently support macOS only")
    if not 1 <= seconds <= 60:
        raise ValueError("Overlay duration must be between 1 and 60 seconds")
    import AppKit
    from Foundation import NSData, NSDate, NSDefaultRunLoopMode

    frame = marker_frame(point, screenshot_size, desktop_size)
    app = AppKit.NSApplication.sharedApplication()
    app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyProhibited)
    screens = AppKit.NSScreen.screens()
    if len(screens) != 1:
        raise RuntimeError("Use a single display for the on-screen overlay")
    screen_frame = screens[0].frame()
    if (screen_frame.size.width, screen_frame.size.height) != desktop_size:
        raise RuntimeError("Display geometry changed; capture again")

    # Render at 2x for Retina. The transparent center leaves the control visible.
    marker = Image.new("RGBA", (160, 160))
    draw = ImageDraw.Draw(marker)
    for color, width in (("black", 10), ("#00eeff", 5)):
        draw.ellipse((36, 36, 124, 124), outline=color, width=width)
        for line in ((80, 12, 80, 46), (80, 114, 80, 148),
                     (12, 80, 46, 80), (114, 80, 148, 80)):
            draw.line(line, fill=color, width=width)
    buffer = io.BytesIO()
    marker.save(buffer, format="PNG")
    data = buffer.getvalue()
    native_image = AppKit.NSImage.alloc().initWithData_(NSData.dataWithBytes_length_(data, len(data)))
    panel = AppKit.NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
        frame, AppKit.NSWindowStyleMaskBorderless | AppKit.NSWindowStyleMaskNonactivatingPanel,
        AppKit.NSBackingStoreBuffered, False)
    panel.setReleasedWhenClosed_(False)
    panel.setOpaque_(False)
    panel.setBackgroundColor_(AppKit.NSColor.clearColor())
    panel.setHasShadow_(False)
    panel.setIgnoresMouseEvents_(True)
    panel.setHidesOnDeactivate_(False)
    panel.setLevel_(AppKit.NSScreenSaverWindowLevel)
    panel.setCollectionBehavior_(AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces |
                                 AppKit.NSWindowCollectionBehaviorFullScreenAuxiliary)
    view = AppKit.NSImageView.alloc().initWithFrame_(((0, 0), (80, 80)))
    view.setImage_(native_image)
    view.setImageScaling_(AppKit.NSImageScaleAxesIndependently)
    panel.setContentView_(view)
    try:
        panel.orderFrontRegardless()
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            event = app.nextEventMatchingMask_untilDate_inMode_dequeue_(
                AppKit.NSEventMaskAny, NSDate.dateWithTimeIntervalSinceNow_(0.05),
                NSDefaultRunLoopMode, True)
            if event is not None:
                app.sendEvent_(event)
            app.updateWindows()
    finally:
        panel.orderOut_(None)
        panel.close()
