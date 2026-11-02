import json
from pathlib import Path
import subprocess
import sys

from flground.evaluation import write_jsonl


def test_suite_dry_run_never_constructs_model_or_creates_output(tmp_path):
    dataset=tmp_path/'data'
    dataset.mkdir()
    write_jsonl(dataset/'tasks.jsonl',[dict(id='1',instruction='target',image='not-needed.png',group='g')])
    output=tmp_path/'run'
    result=subprocess.run([sys.executable,'scripts/benchmark.py','run','--dataset',str(dataset),
                           '--output',str(output)],capture_output=True,text=True,check=True)
    assert '"maximum_calls": 3' in result.stdout
    assert 'normalized_1000' in result.stdout
    assert 'Dry run only' in result.stdout
    assert not output.exists()


def test_suite_rejects_missing_baseline(tmp_path):
    result=subprocess.run([sys.executable,'scripts/benchmark.py','run','--output',str(tmp_path/'out'),
                           '--strategies','panel'],capture_output=True,text=True)
    assert result.returncode != 0
    assert 'baseline first' in result.stderr


def test_reuse_executes_only_candidates_and_filters_full_baseline(tmp_path, monkeypatch):
    import importlib.util
    from PIL import Image
    from flground.evaluation import digest, read_jsonl
    spec=importlib.util.spec_from_file_location('benchmark', 'scripts/benchmark.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data=tmp_path/'data'; data.mkdir()
    Image.new('RGB',(20,20)).save(data/'screen.png')
    task=dict(id='a',instruction='target',image='screen.png',group='g')
    write_jsonl(data/'dev.jsonl',[task])
    write_jsonl(data/'labels.jsonl',[dict(id=i,bbox=[0,0,10,10],width=20,height=20,target_type='text') for i in ['a','b']])
    old=tmp_path/'old'; old.mkdir()
    rows=[dict(id=i,instruction='target',point=dict(x=5,y=5),status='ok',image_size=[20,20],
               image_sha256=digest(data/'screen.png'),latency_seconds=1,calls=[]) for i in ['a','b']]
    write_jsonl(old/'predictions.jsonl',rows)
    (old/'manifest.json').write_text(json.dumps(dict(strategy='baseline',example_ids=['a','b'],model=dict(model='fake',max_tokens=32768))))
    class Fake:
        def manifest(self): return dict(model='fake',max_tokens=4096)
    monkeypatch.setattr(module,'FireworksModel',Fake)
    calls=[]
    def evaluate(tasks,destination,model,strategy,**kwargs):
        calls.append(strategy); destination.mkdir()
        write_jsonl(destination/'predictions.jsonl',rows[:1])
        (destination/'manifest.json').write_text(json.dumps(dict(example_ids=['a'],model=model.manifest())))
    monkeypatch.setattr(module,'evaluate',evaluate)
    output=tmp_path/'out'
    module.run_suite(data,output,'dev',['knowledge','panel_fast'],True,15,4096,old/'predictions.jsonl')
    assert calls==['knowledge','panel_fast']
    assert len(read_jsonl(output/'baseline_reused/predictions.jsonl'))==1
    result=json.loads((output/'summary.json').read_text())
    assert 'UNMATCHED token budgets' in result['panel_fast']['comparison']['compatibility']
    assert len(read_jsonl(old/'predictions.jsonl'))==2
