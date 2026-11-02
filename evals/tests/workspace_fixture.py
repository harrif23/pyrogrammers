import json

import httpx
import pytest
from PIL import Image

from conftest import API_KEY, MODEL
from evals import cli, dataset, evaluate
from evals.baseline import BaselinePredictor
from evals.client import FireworksClient
from evals.reference_qwen import ReferenceQwenPredictor


@pytest.fixture
def workspace(tmp_path, monkeypatch, server):
    data = tmp_path / "data"
    root = data / "screenspot_pro"
    splits = {}
    for split, prefix, count in (("eval", "eval", 3), ("dev", "dev", 8)):
        annotation = root / f"annotations/{split}.json"
        annotation.parent.mkdir(parents=True, exist_ok=True)
        rows, images = [], {}
        for index in range(count):
            name = f"{prefix}/{index}.png"
            path = root / "images" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (200, 100), (index, 1, 2)).save(path)
            rows.append({"id": f"{prefix}_{index}", "img_filename": name, "instruction": "click",
                         "bbox": [90, 40, 110, 60], "img_size": [200, 100],
                         "ui_type": "icon" if index % 2 else "text"})
            images[name] = {"sha256": dataset.sha256_file(path), "width": 200, "height": 100}
        annotation.write_text(json.dumps(rows))
        splits[split] = {"annotation_path": f"annotations/{split}.json",
                         "annotation_sha256": dataset.sha256_file(annotation),
                         "example_ids": [row["id"] for row in rows],
                         "target_counts": {"total": count}, "images": images}
    (data / "manifest.json").write_text(json.dumps({"revision": "abc", "splits": splits}))

    real_load_examples = dataset.load_examples

    def load_manifest():
        return json.loads((data / "manifest.json").read_text())

    def load_examples(manifest, split):
        return real_load_examples(manifest, split, data)

    for module in (cli, evaluate):
        monkeypatch.setattr(module, "load_manifest", load_manifest)
        monkeypatch.setattr(module, "load_examples", load_examples)
        monkeypatch.setattr(module, "load_env_file", lambda: None)
    monkeypatch.setattr(evaluate, "OUTPUTS_DIR", tmp_path / "outputs")
    monkeypatch.setenv("FIREWORKS_API_KEY", API_KEY)
    monkeypatch.setenv("FLGROUND_MODEL", MODEL)

    real_build = evaluate.build_predictor

    def build_predictor(mode, credentials, inference):
        if mode not in ("baseline", "reference"):
            return real_build(mode, credentials, inference)
        http = httpx.Client(transport=httpx.MockTransport(server))
        client = FireworksClient(credentials, inference, http_client=http)
        return BaselinePredictor(client) if mode == "baseline" else ReferenceQwenPredictor(client)

    monkeypatch.setattr(evaluate, "build_predictor", build_predictor)
    return tmp_path
