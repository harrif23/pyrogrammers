import html
from pathlib import Path
from PIL import Image, ImageDraw
from .contracts import GroundingResult


def save_report(image: Image.Image, result: GroundingResult, directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "result.json").write_text(result.model_dump_json(indent=2))
    overlay = image.convert("RGB")
    draw = ImageDraw.Draw(overlay)
    for point, color in [(result.calls[0].point, "cyan"), (result.point, "red")]:
        x, y = point.x, point.y
        draw.ellipse((x-8, y-8, x+8, y+8), outline=color, width=3)
        draw.line((x-12, y, x+12, y), fill=color, width=2)
        draw.line((x, y-12, x, y+12), fill=color, width=2)
    overlay.save(directory / "prediction.png")
    (directory / "report.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>FLGround prediction</title>'
        f'<h1>FLGround: {html.escape(result.strategy)}</h1><p>{html.escape(result.instruction)}</p>'
        '<p>Cyan: first prediction. Red: final prediction.</p>'
        '<img src="prediction.png" style="max-width:100%" alt="Prediction overlay">'
        f'<pre>{html.escape(result.model_dump_json(indent=2))}</pre>')
