"""Offline dataset preparation/scoring, with label-free batch inference inputs."""
import hashlib
import json
import math
import random
import shutil
import subprocess
import time
from pathlib import Path

from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict

from .harness import ground
from .reports import save_report


class Task(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    id: str
    image: str
    instruction: str
    group: str


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    Path(path).write_text(''.join(json.dumps(row) + '\n' for row in rows))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(annotations: Path, image_root: Path, output: Path, revision: str, seed: int = 42):
    """Adapt official ScreenSpot-Pro fields, keeping labels out of tasks."""
    data = json.loads(annotations.read_text())
    if not isinstance(data, list) or not data:
        raise ValueError('Expected nonempty official annotation array')
    tasks, labels, ids = [], [], set()
    for row in data:
        identifier = str(row['id'])
        if identifier in ids:
            raise ValueError('Duplicate example ID')
        ids.add(identifier)
        path = (image_root / row['img_filename']).resolve()
        with Image.open(path) as image:
            width, height = image.size
        if tuple(row['img_size']) != (width, height):
            raise ValueError('Annotation dimensions do not match image')
        box = row['bbox']
        validate_box(box, width, height)
        # Identical screenshot bytes stay together even under different filenames.
        tasks.append(Task(id=identifier, image=str(path), instruction=row['instruction'],
                          group=digest(path)).model_dump())
        labels.append({'id': identifier, 'bbox': box, 'width': width, 'height': height,
                       'target_type': row['ui_type']})
    groups = sorted({row['group'] for row in tasks})
    if len(groups) < 2:
        raise ValueError('Need at least two distinct screenshot groups; use separate development data')
    random.Random(seed).shuffle(groups)
    n = min(len(groups)-1, max(1, round(len(groups)*0.6)))
    development = set(groups[:n])
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output/'tasks.jsonl', tasks)
    write_jsonl(output/'labels.jsonl', labels)
    for split, selected in [('dev', development), ('heldout', set(groups[n:]))]:
        write_jsonl(output/f'{split}.jsonl', [t for t in tasks if t['group'] in selected])
    (output/'dataset.json').write_text(json.dumps({
        'revision': revision, 'annotations_sha256': digest(annotations),
        'tasks_sha256': digest(output/'tasks.jsonl'), 'labels_sha256': digest(output/'labels.jsonl'),
        'seed': seed, 'grouping': 'exact image SHA256; manually merge near-duplicate/session groups before freezing',
        'dev_ids': [t['id'] for t in tasks if t['group'] in development],
        'heldout_ids': [t['id'] for t in tasks if t['group'] not in development],
    }, indent=2))


def validate_box(box, width, height):
    if (not isinstance(box, list) or len(box) != 4 or
            any(type(v) not in (int, float) or not math.isfinite(v) for v in box)):
        raise ValueError('bbox must be four finite pixel coordinates')
    if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
        raise ValueError('Invalid annotation bounds')


def evaluate(tasks_path: Path, output: Path, model, strategy: str, progress=None):
    rows = [Task.model_validate(row) for row in read_jsonl(tasks_path)]
    if not rows or len({r.id for r in rows}) != len(rows):
        raise ValueError('Tasks must have unique IDs and be nonempty')
    output.mkdir(parents=True, exist_ok=False)
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = 'unknown', True
    source_root = Path(__file__).parent
    manifest = {'strategy': strategy, 'tasks_sha256': digest(tasks_path),
                'example_ids': [r.id for r in rows], 'git_revision': revision, 'dirty': dirty,
                'source_sha256': {p.name: digest(p) for p in sorted(source_root.glob('*.py'))},
                'model': model.manifest() if hasattr(model, 'manifest') else {'model': model.model},
                'estimated_cost': None, 'cost_note': 'Supply deployment billing basis; no invented pricing.'}
    dataset_manifest = tasks_path.parent/'dataset.json'
    if dataset_manifest.exists():
        manifest['dataset'] = json.loads(dataset_manifest.read_text())
    snapshot = output/'source'
    snapshot.mkdir()
    for source in source_root.glob('*.py'):
        shutil.copy2(source, snapshot/source.name)
    root = source_root.parent.parent
    for name in ('pyproject.toml', 'uv.lock'):
        if (root/name).exists():
            shutil.copy2(root/name, snapshot/name)
    shutil.copy2(tasks_path, output/'tasks.jsonl')
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    with (output/'predictions.jsonl').open('w') as stream:
        for index, task in enumerate(rows):
            start = time.perf_counter()
            offset = len(getattr(model, 'records', []))
            record = {'id': task.id, 'group': task.group, 'instruction': task.instruction,
                      'point': None, 'status': 'error', 'calls': [], 'error': None}
            try:
                path = Path(task.image)
                if not path.is_absolute():
                    path = tasks_path.parent / path
                record['image_sha256'] = digest(path)
                with Image.open(path) as source:
                    image = source.convert('RGB')
                record['image_size'] = image.size
                destination = output/f'{index:04d}'
                result = ground(image, task.instruction, model, strategy, artifacts=destination)
                save_report(image, result, destination)
                record.update(status=result.status, point=result.point.model_dump() if result.point else None,
                              calls=[c.model_dump() for c in result.calls], metadata=result.metadata)
            except Exception as exc:
                # Exception strings from SDKs can contain request data; save category only.
                record['error'] = type(exc).__name__
            finally:
                if hasattr(model, 'records'):
                    record['calls'] = [c.model_dump() for c in model.records[offset:]]
                record['latency_seconds'] = time.perf_counter()-start
                record['model_calls'] = len(record['calls'])
                stream.write(json.dumps(record)+'\n')
                stream.flush()
                if progress is not None:
                    progress(index+1, len(rows), record)


def unique(rows):
    result = {r['id']: r for r in rows}
    if len(result) != len(rows):
        raise ValueError('Duplicate IDs')
    return result


def score(predictions: Path, labels_path: Path, baseline: Path | None = None):
    predictions_by_id = unique(read_jsonl(predictions))
    if not predictions_by_id:
        raise ValueError('No predictions')
    labels = unique(read_jsonl(labels_path))
    if not predictions_by_id.keys() <= labels.keys():
        raise ValueError('Missing labels')
    manifest_path = predictions.parent/'manifest.json'
    if manifest_path.exists():
        expected = json.loads(manifest_path.read_text())['example_ids']
        if set(expected) != predictions_by_id.keys():
            raise ValueError('Incomplete run: prediction IDs do not match manifest')

    def correct(row, label):
        validate_box(label['bbox'], label['width'], label['height'])
        if row.get('status') != 'ok' or row.get('point') is None:
            return False
        if tuple(row.get('image_size', ())) != (label['width'], label['height']):
            raise ValueError('Prediction/label dimensions differ')
        x, y = row['point']['x'], row['point']['y']
        l, t, r, b = label['bbox']
        return (type(x) in (float, int) and type(y) in (float, int) and
                math.isfinite(x) and math.isfinite(y) and
                0 <= x < label['width'] and 0 <= y < label['height'] and l <= x <= r and t <= y <= b)

    outcomes = {i: correct(row, labels[i]) for i, row in predictions_by_id.items()}
    latencies = sorted(r['latency_seconds'] for r in predictions_by_id.values())
    usage = {}
    for row in predictions_by_id.values():
        for call in row.get('calls', []):
            for key, value in call.get('usage', {}).items():
                if type(value) in (int, float):
                    usage[key] = usage.get(key, 0)+value
    n = len(outcomes)
    report = {'correct': sum(outcomes.values()), 'total': n, 'accuracy': sum(outcomes.values())/n,
              'errors': sum(r['status'] == 'error' for r in predictions_by_id.values()),
              'abstentions': sum(r['status'] == 'not_found' for r in predictions_by_id.values()),
              'mean_latency_seconds': sum(latencies)/n,
              'p95_latency_seconds': latencies[math.ceil(0.95*n)-1],
              'model_calls': sum(r.get('model_calls', len(r.get('calls', []))) for r in predictions_by_id.values()),
              'usage': usage, 'slices': {}, 'estimated_cost': None}
    for target_type in sorted({labels[i]['target_type'] for i in outcomes}):
        values = [ok for i, ok in outcomes.items() if labels[i]['target_type'] == target_type]
        report['slices'][target_type] = {'correct': sum(values), 'total': len(values)}
    if baseline:
        old = unique(read_jsonl(baseline))
        if old.keys() != predictions_by_id.keys():
            raise ValueError('Paired comparison requires identical example IDs')
        for identifier, row in predictions_by_id.items():
            if (old[identifier]['instruction'] != row['instruction'] or
                    old[identifier].get('image_sha256') != row.get('image_sha256')):
                raise ValueError('Paired comparison requires identical instructions/images')
        old_manifest_path = baseline.parent/'manifest.json'
        compatibility = 'unverified: one or both manifests missing'
        if manifest_path.exists() and old_manifest_path.exists():
            current_manifest = json.loads(manifest_path.read_text())
            old_manifest = json.loads(old_manifest_path.read_text())
            if current_manifest['model'] != old_manifest['model']:
                raise ValueError('Model/settings differ; not a matched-backbone comparison')
            compatibility = 'model/settings matched; inspect prompts and source hashes in manifests'
        before = {i: correct(old[i], labels[i]) for i in outcomes}
        fixes = [i for i in outcomes if outcomes[i] and not before[i]]
        regressions = [i for i in outcomes if before[i] and not outcomes[i]]
        report['comparison'] = {'compatibility': compatibility, 'baseline_correct': sum(before.values()), 'fixes': fixes,
                                'regressions': regressions, 'net_fixes': len(fixes)-len(regressions),
                                'delta_percentage_points': 100*(len(fixes)-len(regressions))/n}
    return report


def calibration(output: Path):
    """Independent synthetic fixtures; no benchmark labels or paid calls."""
    output.mkdir(parents=True, exist_ok=False)
    tasks, labels = [], []
    for index, (width, height) in enumerate([(800, 600), (1512, 982), (2400, 1600)]):
        image = Image.new('RGB', (width, height), '#222222')
        draw = ImageDraw.Draw(image)
        for name, fx, fy, color in [('red', .15, .2, '#ff3333'), ('green', .75, .3, '#33ff33'), ('blue', .6, .8, '#3333ff')]:
            x, y = int(width*fx), int(height*fy)
            box = [x-30, y-30, x+30, y+30]
            draw.rectangle(box, fill=color)
            identifier = f'{index}-{name}'
            tasks.append({'id': identifier, 'image': f'{index}.png',
                          'instruction': f'Find the center of the {name} square', 'group': str(index)})
            labels.append({'id': identifier, 'bbox': box, 'width': width, 'height': height, 'target_type': 'icon'})
        image.save(output/f'{index}.png')
    write_jsonl(output/'tasks.jsonl', tasks)
    write_jsonl(output/'labels.jsonl', labels)
