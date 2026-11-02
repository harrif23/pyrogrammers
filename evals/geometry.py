"""Box and point validation plus coordinate conversion, shared by every strategy."""


def box_problem(bbox, width, height):
    """Why a gold [x1, y1, x2, y2] box is unusable for this image size, or None if it is fine."""
    x1, y1, x2, y2 = bbox
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        return f"bbox {list(bbox)} invalid for {width}x{height}"
    return None


def normalized_to_pixels(point, width, height):
    """Normalized (top-left origin) to original-image pixels."""
    return [point[0] * width, point[1] * height]
