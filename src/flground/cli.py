from enum import Enum
from pathlib import Path
from uuid import uuid4
import typer
from PIL import Image, ImageChops
from .harness import ground
from .model import FireworksModel
from .reports import save_report

app = typer.Typer(no_args_is_help=True)


class Strategy(str, Enum):
    baseline = "baseline"
    refine = "refine"


def run(image, instruction, strategy, output):
    destination = output / uuid4().hex
    result = ground(image, instruction, FireworksModel(), strategy.value, artifacts=destination)
    save_report(image, result, destination)
    typer.echo(f"({result.point.x:.1f}, {result.point.y:.1f}) — {destination / 'report.html'}")
    return result


@app.command()
def predict(image: Path = typer.Option(..., exists=True, dir_okay=False),
            instruction: str = typer.Option(...), strategy: Strategy = Strategy.baseline,
            output: Path = Path("artifacts")):
    with Image.open(image) as source:
        run(source.convert("RGB"), instruction, strategy, output)


@app.command()
def demo(instruction: str = typer.Option(...), strategy: Strategy = Strategy.baseline,
         execute_click: bool = False, output: Path = Path("artifacts")):
    """Capture macOS desktop; only click when explicitly requested and confirmed."""
    from .desktop_macos import capture, click
    image, desktop = capture()
    result = run(image, instruction, strategy, output)
    if execute_click and typer.confirm("Inspect the saved overlay. Execute this click?"):
        fresh, current_desktop = capture()
        if fresh.size != image.size or current_desktop != desktop or ImageChops.difference(image, fresh).getbbox():
            raise typer.BadParameter("Screen changed since prediction; rerun before clicking")
        click(result.point, image.size, desktop)


if __name__ == "__main__":
    app()
