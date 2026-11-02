import json
import shutil

import pytest
from PIL import Image

from evals import dataset
from evals.errors import DatasetError


def row(example_id, filename, ui_type="icon", bbox=(1, 1, 5, 5), size=(20, 10)):
    return {"id": example_id, "img_filename": filename, "instruction": "click", "bbox": list(bbox),
            "img_size": list(size), "ui_type": ui_type}


@pytest.fixture
def fake_hub(tmp_path):
    """A local stand-in for the hub that records every path requested."""
    hub = tmp_path / "hub"
    annotations = {
        dataset.EVAL_ANNOTATION: [row("eval_0", "eval/a.png"), row("eval_1", "eval/b.png", "text")],
        dataset.DEV_ANNOTATION: [row(f"dev_{i}", f"dev/{i}.png", "icon" if i % 2 else "text")
                                 for i in range(12)],
    }
    for name, rows in annotations.items():
        path = hub / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows))
        for index, item in enumerate(rows):
            image = hub / "images" / item["img_filename"]
            image.parent.mkdir(parents=True, exist_ok=True)
            # Unique pixels, so the content-overlap check isn't tripped by identical files.
            blue = 7 if name == dataset.EVAL_ANNOTATION else 200
            Image.new("RGB", (20, 10), (1, index, blue)).save(image)
    requested = []

    def download(path_in_repo, revision, data_dir):
        requested.append(path_in_repo)
        target = data_dir / "screenspot_pro" / path_in_repo
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(hub / path_in_repo, target)
        return target

    download.requested = requested
    download.hub = hub
    return download


def test_manifest_records_the_pin_and_counts(tmp_path, fake_hub):
    manifest = dataset.prepare(tmp_path / "data", revision="abc", download=fake_hub)
    entry = manifest["splits"]["eval"]
    assert manifest["revision"] == "abc"
    assert entry["example_ids"] == ["eval_0", "eval_1"]
    assert entry["target_counts"] == {"total": 2, "icon": 1, "text": 1}
    assert entry["unique_screenshot_count"] == 2
    assert len(entry["annotation_sha256"]) == 64
    assert manifest["eval_matches_expected"] is False


def test_only_referenced_screenshots_are_downloaded(tmp_path, fake_hub):
    dataset.prepare(tmp_path / "data", revision="abc", download=fake_hub)
    images = [path for path in fake_hub.requested if path.startswith("images/")]
    assert len(images) == 2 + dataset.DEV_POOL_SIZE
    assert "images/dev/11.png" not in images


def test_existing_pin_is_kept_and_cannot_silently_move(tmp_path, fake_hub):
    data = tmp_path / "data"
    dataset.prepare(data, revision="abc", download=fake_hub)
    assert dataset.prepare(data, download=fake_hub)["revision"] == "abc"
    with pytest.raises(DatasetError):
        dataset.prepare(data, revision="other", download=fake_hub)


def test_dev_screenshot_overlap_is_refused(tmp_path, fake_hub):
    shutil.copy(fake_hub.hub / "images/eval/a.png", fake_hub.hub / "images/dev/0.png")
    with pytest.raises(DatasetError, match="overlap"):
        dataset.prepare(tmp_path / "data", revision="abc", download=fake_hub)
    assert not (tmp_path / "data" / "manifest.json").exists()


def test_image_size_must_match_its_annotation(tmp_path, fake_hub):
    Image.new("RGB", (40, 10)).save(fake_hub.hub / "images/eval/a.png")
    with pytest.raises(DatasetError, match="annotation says"):
        dataset.prepare(tmp_path / "data", revision="abc", download=fake_hub)


def test_changed_image_is_detected_at_load(tmp_path, fake_hub):
    data = tmp_path / "data"
    manifest = dataset.prepare(data, revision="abc", download=fake_hub)
    assert len(dataset.load_examples(manifest, "eval", data)) == 2
    Image.new("RGB", (20, 10), (1, 2, 3)).save(data / "screenspot_pro/images/eval/a.png")
    with pytest.raises(DatasetError, match="changed"):
        dataset.load_examples(manifest, "eval", data)


def test_changed_annotation_is_detected_at_load(tmp_path, fake_hub):
    data = tmp_path / "data"
    manifest = dataset.prepare(data, revision="abc", download=fake_hub)
    (data / "screenspot_pro" / dataset.EVAL_ANNOTATION).write_text("[]")
    with pytest.raises(DatasetError, match="changed"):
        dataset.load_examples(manifest, "eval", data)


@pytest.mark.parametrize("bad", [
    row("a", "x.png", bbox=(5, 1, 1, 5)),
    row("a", "x.png", bbox=(1, 1, 21, 5)),
    row("a", "x.png", ui_type="button"),
    {**row("a", "x.png"), "instruction": " "},
    {key: value for key, value in row("a", "x.png").items() if key != "bbox"},
])
def test_invalid_annotations_are_rejected(bad):
    with pytest.raises(DatasetError):
        dataset.validate_annotations([bad])


def test_duplicate_ids_are_rejected():
    with pytest.raises(DatasetError, match="duplicate"):
        dataset.validate_annotations([row("a", "x.png"), row("a", "y.png")])


def test_dev_pool_is_deterministic_and_alternates_types():
    rows = [row(f"d_{i}", f"{i}.png", "icon" if i % 2 else "text") for i in range(20)]
    selected = dataset.select_dev_rows(rows, size=4)
    assert [item["ui_type"] for item in selected] == ["icon", "text", "icon", "text"]
    assert [item["id"] for item in selected] == ["d_1", "d_0", "d_3", "d_2"]
