"""Pinned download, validation, and the manifest for benchmark data."""

import base64
import collections
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from PIL import Image

from .config import EVALS_DIR
from .errors import DatasetError
from .geometry import box_problem
from .client import MAX_BASE64_BYTES

REPO_ID = "likaixin/ScreenSpot-Pro"
EVAL_ANNOTATION = "annotations/fruitloops_windows.json"
# why: same resolution and platform as FL Studio but a different application, so prompt
# and budget tuning on it cannot leak the final evaluation set.
DEV_ANNOTATION = "annotations/premiere_windows.json"
DEV_POOL_SIZE = 8
EXPECTED_TARGETS = {"total": 57, "icon": 31, "text": 26}

DATA_DIR = EVALS_DIR / "data"
MANIFEST_PATH = DATA_DIR / "manifest.json"


@dataclass(frozen=True)
class Example:
    id: str
    image_path: Path
    instruction: str
    bbox: tuple[float, float, float, float]
    width: int
    height: int
    ui_type: str


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_annotations(rows):
    """Raises DatasetError unless every row is usable by the scorer."""
    ids = [row.get("id") for row in rows]
    duplicates = [key for key, count in collections.Counter(ids).items() if count > 1]
    if duplicates:
        raise DatasetError(f"duplicate example ids: {duplicates}")
    for row in rows:
        example_id = row.get("id")
        missing = {"id", "img_filename", "instruction", "bbox", "img_size", "ui_type"} - set(row)
        if missing:
            raise DatasetError(f"{example_id}: missing fields {sorted(missing)}")
        if not str(row["instruction"]).strip():
            raise DatasetError(f"{example_id}: empty instruction")
        if row["ui_type"] not in ("icon", "text"):
            raise DatasetError(f"{example_id}: unknown ui_type {row['ui_type']!r}")
        problem = box_problem(row["bbox"], *row["img_size"])
        if problem:
            raise DatasetError(f"{example_id}: {problem}")


def target_counts(rows):
    counts = collections.Counter(row["ui_type"] for row in rows)
    return {"total": len(rows), "icon": counts["icon"], "text": counts["text"]}


def select_dev_rows(rows, size=DEV_POOL_SIZE):
    """Deterministic pool alternating icon and text, so small pilots see both types."""
    def numeric_id(row):
        match = re.search(r"(\d+)$", row["id"])
        return (int(match.group(1)) if match else -1, row["id"])

    icons = sorted((r for r in rows if r["ui_type"] == "icon"), key=numeric_id)
    texts = sorted((r for r in rows if r["ui_type"] == "text"), key=numeric_id)
    selected = []
    for pair in zip(icons, texts):
        selected.extend(pair)
    return selected[:size]


def _download(path_in_repo, revision, data_dir):
    return Path(hf_hub_download(REPO_ID, path_in_repo, repo_type="dataset", revision=revision,
                                local_dir=data_dir / "screenspot_pro"))


def _describe_images(rows, revision, data_dir, download):
    """Fetch only the referenced screenshots and verify them against their annotations."""
    images = {}
    for row in rows:
        filename = row["img_filename"]
        path = download(f"images/{filename}", revision, data_dir)
        with Image.open(path) as opened:
            size = list(opened.size)
        if size != list(row["img_size"]):
            raise DatasetError(f"{row['id']}: image is {size}, annotation says {row['img_size']}")
        raw = path.read_bytes()
        base64_bytes = len(base64.b64encode(raw))
        if base64_bytes >= MAX_BASE64_BYTES:
            raise DatasetError(f"{filename}: too large to send without resizing")
        images[filename] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "width": size[0],
            "height": size[1],
            "bytes": len(raw),
            "base64_bytes": base64_bytes,
        }
    return images


def _prepare_split(annotation_path, select, revision, data_dir, download):
    path = download(annotation_path, revision, data_dir)
    rows = json.loads(path.read_text())
    validate_annotations(rows)
    rows = select(rows)
    return {
        "annotation_path": annotation_path,
        "annotation_sha256": sha256_file(path),
        "example_ids": [row["id"] for row in rows],
        "target_counts": target_counts(rows),
        "unique_screenshot_count": len({row["img_filename"] for row in rows}),
        "images": _describe_images(rows, revision, data_dir, download),
    }


def resolve_revision(manifest_path=MANIFEST_PATH, requested=None):
    """An existing manifest's pin is kept, so re-running prepare cannot move the benchmark."""
    manifest_path = Path(manifest_path)
    pinned = json.loads(manifest_path.read_text())["revision"] if manifest_path.exists() else None
    if requested and pinned and requested != pinned:
        raise DatasetError(f"manifest is pinned to {pinned}; move it aside to change revision")
    return requested or pinned or HfApi().dataset_info(REPO_ID).sha


def prepare(data_dir=DATA_DIR, revision=None, download=_download):
    """Download and validate both splits, then write the manifest. Makes no model calls.

    rules: eval and dev must share no screenshot, by filename or content.
    side effects: writes under data_dir only.
    raises: DatasetError on any validation failure, before a manifest is written.
    """
    data_dir = Path(data_dir)
    manifest_path = data_dir / "manifest.json"
    revision = resolve_revision(manifest_path, revision)
    evaluation = _prepare_split(EVAL_ANNOTATION, lambda rows: rows, revision, data_dir, download)
    dev = _prepare_split(DEV_ANNOTATION, select_dev_rows, revision, data_dir, download)

    shared = set(evaluation["images"]) & set(dev["images"])
    shared_content = ({i["sha256"] for i in evaluation["images"].values()}
                      & {i["sha256"] for i in dev["images"].values()})
    if shared or shared_content:
        raise DatasetError("development screenshots overlap the evaluation set")

    manifest = {
        "dataset": REPO_ID,
        "revision": revision,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "expected_eval_targets": EXPECTED_TARGETS,
        "eval_matches_expected": evaluation["target_counts"] == EXPECTED_TARGETS,
        "splits": {"eval": evaluation, "dev": dev},
    }
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_manifest(manifest_path=MANIFEST_PATH):
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise DatasetError("no manifest found; run `evals prepare` first")
    return json.loads(manifest_path.read_text())


def load_examples(manifest, split, data_dir=DATA_DIR):
    """Load a split, re-verifying annotation and image hashes against the manifest.

    raises: DatasetError if any file changed since prepare, which would invalidate the pin.
    """
    data_dir = Path(data_dir)
    entry = manifest["splits"][split]
    root = data_dir / "screenspot_pro"
    annotation = root / entry["annotation_path"]
    if not annotation.exists() or sha256_file(annotation) != entry["annotation_sha256"]:
        raise DatasetError(f"{annotation} is missing or changed; re-run `evals prepare`")
    by_id = {row["id"]: row for row in json.loads(annotation.read_text())}

    examples = []
    for example_id in entry["example_ids"]:
        row = by_id[example_id]
        path = root / "images" / row["img_filename"]
        info = entry["images"][row["img_filename"]]
        if not path.exists() or sha256_file(path) != info["sha256"]:
            raise DatasetError(f"{path} is missing or changed; re-run `evals prepare`")
        examples.append(Example(
            id=example_id,
            image_path=path,
            instruction=row["instruction"],
            bbox=tuple(row["bbox"]),
            width=info["width"],
            height=info["height"],
            ui_type=row["ui_type"],
        ))
    return examples
