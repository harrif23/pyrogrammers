from enum import Enum
from pathlib import Path
import time
import json
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
    knowledge = "knowledge"
    panel = "panel"
    panel_fast = "panel_fast"
    panel_recover = "panel_recover"
    panel_compact = "panel_compact"
    panel_selective = "panel_selective"
    panel_plain = "panel_plain"
    panel_verify = "panel_verify"


def run(image, instruction, strategy, output):
    destination = output / uuid4().hex
    model = FireworksModel()
    destination.mkdir(parents=True, exist_ok=True)
    try:
        result = ground(image, instruction, model, strategy.value, artifacts=destination)
    except Exception as exc:
        (destination / "failure.json").write_text(json.dumps({
            "status": "error", "error": type(exc).__name__, "instruction": instruction,
            "strategy": strategy.value, "model": model.manifest(),
            "calls": [c.model_dump() for c in model.records]}, indent=2))
        raise typer.BadParameter(f"Inference failed ({type(exc).__name__}); details: {destination / 'failure.json'}") from None
    save_report(image, result, destination)
    if result.point is None:
        typer.echo(f"Target not identified — {destination / 'report.html'}")
        return result
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
         execute_click: bool = False, output: Path = Path("artifacts"),
         delay: int = typer.Option(5, min=0, max=60, help="Seconds to switch to FL Studio before capture."),
         overlay: bool = typer.Option(True, help="Show a temporary click-through target marker."),
         overlay_seconds: int = typer.Option(10, min=1, max=60, help="How long the marker stays visible.")):
    """Capture macOS desktop and mark the predicted control over the app."""
    from .desktop_macos import capture, click
    if delay:
        typer.echo(f"Switch to FL Studio now. Capturing the whole display in {delay} seconds.")
        time.sleep(delay)
    image, desktop = capture()
    typer.echo("Captured. Locating the control; keep the window and controls in place...")
    result = run(image, instruction, strategy, output)
    if result.point is None:
        return
    if overlay:
        from .overlay_macos import show_overlay
        typer.echo(f"Showing target for {overlay_seconds} seconds. Marker refers to the captured screen.")
        show_overlay(result.point, image.size, desktop, overlay_seconds)
    if execute_click and typer.confirm("Inspect the saved overlay. Execute this click?"):
        fresh, current_desktop = capture()
        if fresh.size != image.size or current_desktop != desktop or ImageChops.difference(image, fresh).getbbox():
            raise typer.BadParameter("Screen changed since prediction; rerun before clicking")
        click(result.point, image.size, desktop)


@app.command("prepare-benchmark")
def prepare_benchmark(annotations: Path, image_root: Path, output: Path,
                      revision: str = typer.Option(...), seed: int = 42):
    """Prepare official annotation JSON into separate tasks, labels and grouped splits. No API calls."""
    from .evaluation import prepare
    prepare(annotations, image_root, output, revision, seed)
    typer.echo(f"Prepared {output}. Review near-duplicate groups before freezing splits.")


@app.command()
def calibrate(output: Path):
    """Generate synthetic coordinate-calibration fixtures. No API calls."""
    from .evaluation import calibration
    calibration(output)
    typer.echo(f"Fixtures: {output / 'tasks.jsonl'}")


@app.command()
def evaluate(tasks: Path, output: Path, strategy: Strategy = Strategy.baseline):
    """Run billed inference over label-free task JSONL; persist every outcome."""
    from .evaluation import evaluate as evaluate_batch
    evaluate_batch(tasks, output, FireworksModel(), strategy.value)
    typer.echo(f"Predictions: {output / 'predictions.jsonl'}")


@app.command()
def score(predictions: Path, labels: Path, baseline: Path | None = None):
    """Score saved predictions offline; optionally report paired fixes/regressions."""
    from .evaluation import score as score_batch
    typer.echo(json.dumps(score_batch(predictions, labels, baseline), indent=2))


if __name__ == "__main__":
    app()
