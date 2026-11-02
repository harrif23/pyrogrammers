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
