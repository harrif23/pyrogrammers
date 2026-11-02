import json
from pathlib import Path

import pytest
from PIL import Image
from typer.testing import CliRunner

from flground.cli import app
from flground.contracts import CallRecord, Point
from flground.evaluation import calibration, evaluate, prepare, read_jsonl, score, write_jsonl


class Fake:
    model = 'fake'

    def predict(self, image, instruction):
        assert instruction in ('pass','fail')
        if instruction == 'fail':
            raise ValueError('bad output')
        return CallRecord(point=Point(x=10,y=20), raw_response='{}', latency_seconds=.01)


def test_batch_errors_count_and_labels_never_enter_inference(tmp_path):
    Image.new('RGB',(100,80)).save(tmp_path/'screen.png')
    tasks = [dict(id=str(i),image='screen.png',instruction=text,group='g') for i,text in enumerate(['pass','fail'])]
    write_jsonl(tmp_path/'tasks.jsonl',tasks)
    labels = [dict(id=str(i), bbox=[10,20,30,40],width=100,height=80,target_type='icon') for i in range(2)]
    write_jsonl(tmp_path/'labels.jsonl',labels)
    evaluate(tmp_path/'tasks.jsonl',tmp_path/'run',Fake(),'baseline')
    predictions = tmp_path/'run/predictions.jsonl'
    rows = read_jsonl(predictions)
    assert [r['status'] for r in rows] == ['ok','error']
    summary = score(predictions,tmp_path/'labels.jsonl')
    assert summary['correct'] == 1 and summary['total'] == 2 and summary['errors'] == 1
    assert summary['accuracy'] == .5
    # Partial run must not silently shrink denominator.
    write_jsonl(predictions,rows[:1])
    with pytest.raises(ValueError, match='Incomplete'):
        score(predictions,tmp_path/'labels.jsonl')


def test_label_fields_in_task_rejected_before_inference(tmp_path):
    write_jsonl(tmp_path/'tasks.jsonl',[dict(id='1',image='a',instruction='pass',group='g',bbox=[1,2,3,4])])
    with pytest.raises(ValueError):
        evaluate(tmp_path/'tasks.jsonl',tmp_path/'run',Fake(),'baseline')
    assert not (tmp_path/'run').exists()


def test_paired_fixes_and_regressions(tmp_path):
    labels = [dict(id=str(i),bbox=[10,10,20,20],width=100,height=100,target_type='icon') for i in range(3)]
    write_jsonl(tmp_path/'labels', labels)
    def rows(points):
        return [dict(id=str(i),status='ok',point=dict(x=x,y=x),image_size=[100,100],
                     instruction='same',image_sha256='same',latency_seconds=1,calls=[]) for i,x in enumerate(points)]
    write_jsonl(tmp_path/'before',rows([15,0,0]))
    write_jsonl(tmp_path/'after',rows([0,15,15]))
    result = score(tmp_path/'after',tmp_path/'labels',tmp_path/'before')
    assert result['comparison']['fixes'] == ['1','2']
    assert result['comparison']['regressions'] == ['0']
    assert result['comparison']['net_fixes'] == 1


def test_prepare_groups_duplicate_screens_and_separates_labels(tmp_path):
    rows=[]
    for i,color in enumerate(['black','black','white']):
        Image.new('RGB',(100,80),color).save(tmp_path/f'{i}.png')
        rows.append(dict(id=str(i),img_filename=f'{i}.png',instruction='pass',bbox=[10,10,20,20],
                         img_size=[100,80],ui_type='icon'))
    (tmp_path/'official.json').write_text(json.dumps(rows))
    prepare(tmp_path/'official.json',tmp_path,tmp_path/'dataset','revision-test')
    dev=read_jsonl(tmp_path/'dataset/dev.jsonl')
    heldout=read_jsonl(tmp_path/'dataset/heldout.jsonl')
    assert {r['group'] for r in dev}.isdisjoint({r['group'] for r in heldout})
    assert all(set(r)=={'id','image','instruction','group'} for r in dev+heldout)
    assert next(r for r in dev+heldout if r['id']=='0')['group'] == next(r for r in dev+heldout if r['id']=='1')['group']


def test_calibration_offline_fixtures(tmp_path):
    calibration(tmp_path/'fixtures')
    tasks=read_jsonl(tmp_path/'fixtures/tasks.jsonl')
    assert len(tasks)==9
    assert all((tmp_path/'fixtures'/r['image']).is_file() for r in tasks)


def test_demo_abstention_does_not_overlay_or_click(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr('flground.desktop_macos.capture',lambda: (Image.new('RGB',(100,80)),(100,80)))
    monkeypatch.setattr('flground.cli.run',lambda *args: SimpleNamespace(point=None))
    def forbidden(*args):
        raise AssertionError('Must not overlay or click')
    monkeypatch.setattr('flground.overlay_macos.show_overlay',forbidden)
    monkeypatch.setattr('flground.desktop_macos.click',forbidden)
    result=CliRunner().invoke(app,['demo','--instruction','missing','--delay','0','--execute-click'])
    assert result.exit_code==0,result.output


def test_parallel_calls_are_isolated_and_errors_persist(tmp_path):
    from threading import Barrier
    barrier = Barrier(3)
    class ConcurrentFake:
        model = 'fake'
        def __init__(self):
            self.records = []
        def predict(self, image, instruction):
            barrier.wait(timeout=5)
            call = CallRecord(point=Point(x=10,y=20), raw_response=instruction, latency_seconds=.01)
            self.records.append(call)
            if instruction == 'fail':
                raise ValueError('bad output')
            return call
    Image.new('RGB',(100,80)).save(tmp_path/'screen.png')
    write_jsonl(tmp_path/'tasks.jsonl', [dict(id=str(i), image='screen.png', instruction=t, group='g')
                                     for i,t in enumerate(['one','two','fail'])])
    evaluate(tmp_path/'tasks.jsonl',tmp_path/'run',ConcurrentFake(),'baseline',
             concurrency=3,model_factory=ConcurrentFake)
    rows=read_jsonl(tmp_path/'run/predictions.jsonl')
    assert len(rows)==3
    assert sum(r['status']=='error' for r in rows)==1
    for row in rows:
        assert len(row['calls'])==1
        assert row['calls'][0]['raw_response']==row['instruction']
    manifest=json.loads((tmp_path/'run/manifest.json').read_text())
    assert manifest['concurrency']==3 and manifest['wall_seconds']>0


def test_parallel_factory_failure_is_counted(tmp_path):
    write_jsonl(tmp_path/'tasks.jsonl',[dict(id='1',image='unused',instruction='pass',group='g')])
    def broken():
        raise ValueError('setup failed')
    evaluate(tmp_path/'tasks.jsonl',tmp_path/'run',Fake(),'baseline',concurrency=2,model_factory=broken)
    assert read_jsonl(tmp_path/'run/predictions.jsonl')[0]['status']=='error'
