from dataclasses import dataclass
from .contracts import Point


def validate_point(point: Point, width: int, height: int) -> Point:
    if not (0 <= point.x < width and 0 <= point.y < height):
        raise ValueError(f"Prediction ({point.x}, {point.y}) outside {width}x{height} image")
    return point


@dataclass(frozen=True)
class Transform:
    """Map coordinates in a resized crop back to its source image."""
    left: int
    top: int
    source_width: int
    source_height: int
    output_width: int
    output_height: int

    def __post_init__(self):
        if min(self.source_width, self.source_height, self.output_width, self.output_height) <= 0:
            raise ValueError("Transform dimensions must be positive")

    def to_source(self, point: Point) -> Point:
        validate_point(point, self.output_width, self.output_height)
        return Point(x=self.left + point.x * self.source_width / self.output_width,
                     y=self.top + point.y * self.source_height / self.output_height)


def crop_bounds(point: Point, width: int, height: int, size: int) -> tuple[int, int, int, int]:
    validate_point(point, width, height)
    if size <= 0:
        raise ValueError("Crop size must be positive")
    w, h = min(size, width), min(size, height)
    left = max(0, min(int(point.x) - w // 2, width - w))
    top = max(0, min(int(point.y) - h // 2, height - h))
    return left, top, left + w, top + h
