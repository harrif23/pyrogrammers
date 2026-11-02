"""Inspection images drawn from a finished results log. Never used during inference."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

GOLD_COLOR = (0, 230, 90)
PREDICTION_COLOR = (255, 40, 40)
_CROSS_RADIUS = 22
_LINE_WIDTH = 3
_BANNER_HEIGHT = 34


def draw_overlay(image_path, record, output_path):
    """Save a copy of the screenshot with the gold box, predicted point, and a verdict banner."""
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle(record["gold_bbox"], outline=GOLD_COLOR, width=_LINE_WIDTH)

    point = record.get("point_px")
    if point:
        x, y = point
        draw.line([(x - _CROSS_RADIUS, y), (x + _CROSS_RADIUS, y)],
                  fill=PREDICTION_COLOR, width=_LINE_WIDTH)
        draw.line([(x, y - _CROSS_RADIUS), (x, y + _CROSS_RADIUS)],
                  fill=PREDICTION_COLOR, width=_LINE_WIDTH)
        verdict = "CORRECT" if record["correct"] else "WRONG"
    else:
        verdict = f"INVALID ({record.get('invalid_reason')})"

    banner = f"{record['example_id']} [{record['ui_type']}] {verdict}: {record['instruction']}"
    # Added canvas, because targets can sit in the menu bar a drawn-over banner would hide.
    canvas = Image.new("RGB", (image.width, image.height + _BANNER_HEIGHT), (0, 0, 0))
    canvas.paste(image, (0, _BANNER_HEIGHT))
    ImageDraw.Draw(canvas).text((8, 6), banner, fill=(255, 255, 255),
                                font=ImageFont.load_default(size=20))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def draw_run_overlays(records, examples, output_dir):
    """Draw the latest scored attempt per example. API-error examples have nothing to draw."""
    paths = {example.id: example.image_path for example in examples}
    latest = {r["example_id"]: r for r in records if r["status"] == "scored"}
    written = []
    for example_id, record in latest.items():
        output_path = Path(output_dir) / f"{example_id}.png"
        draw_overlay(paths[example_id], record, output_path)
        written.append(output_path)
    return written
