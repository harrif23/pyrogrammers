import json

import pytest
from PIL import Image

from flground.contracts import CallRecord, Point
from flground.harness import ground
from flground.model import to_pixels
from flground.panel import Box


def plan(**updates):
    value = dict(visible=True, panel='Channel Rack', entity='Snare', control='volume',
                 evidence='Snare label beside two knobs',
                 region=dict(left=100, top=100, right=300, bottom=200),
                 anchor=dict(left=80, top=120, right=100, bottom=140), point=dict(x=200,y=150))
    value.update(updates)
    return value


class StructuredFake:
    model = 'fake'
    coordinate_space = 'pixels'

    def __init__(self, responses):
        self.responses = iter(responses)
        self.records = []
        self.images = []

    def complete(self, images, prompt, stage):
        self.images.append([im.size for im in images])
        raw = next(self.responses)
        if isinstance(raw, Exception):
            raise raw
        record = CallRecord(raw_response=json.dumps(raw), latency_seconds=.1, prompt=prompt, stage=stage)
        self.records.append(record)
        return record


def test_panel_retains_anchor_and_maps_crop(tmp_path):
    model = StructuredFake([plan(), {'x':100, 'y':80}])
    result = ground(Image.new('RGB', (1000,800)), 'Snare channel volume', model, 'panel', artifacts=tmp_path)
    assert result.point == Point(x=119, y=129)
    assert result.model_calls == 2
    assert result.metadata['crop']['bounds'] == (69,89,311,211)
    assert model.images == [[(1000,800)], [(484,244)]]
    assert (tmp_path/'refinement.png').exists()
    assert 'Channel Rack rows' in model.records[0].prompt
    assert 'ground_truth' not in model.records[0].prompt


@pytest.mark.parametrize('choice,expected', [('initial', Point(x=200,y=150)),
                                           ('refined', Point(x=119,y=129)), ('neither', None)])
def test_verifier_can_reject_refinement(choice,expected):
    model = StructuredFake([plan(), {'x':100,'y':80}, {'choice':choice, 'evidence':'visible match'}])
    result = ground(Image.new('RGB',(1000,800)), 'Snare volume', model, 'panel_verify')
    assert result.point == expected
    assert result.status == ('ok' if expected else 'not_found')
    assert result.model_calls == 3
    assert model.images[-1] == [(1000,800),(484,244),(484,244)]


def test_absent_target_stops_after_one_call():
    model = StructuredFake([plan(visible=False, region=None, anchor=None, point=None)])
    result = ground(Image.new('RGB',(1000,800)), 'Snare volume', model, 'panel_verify')
    assert result.point is None and result.model_calls == 1


def test_local_absence_never_verifies_or_guesses():
    model = StructuredFake([plan(), {'visible':False}])
    result = ground(Image.new('RGB',(1000,800)), 'Snare volume', model, 'panel_verify')
    assert result.point is None and result.model_calls == 2


@pytest.mark.parametrize('bad', [plan(region=dict(left=-1,top=0,right=300,bottom=200)),
                               plan(point=dict(x=900,y=700)), plan(point=None)])
def test_invalid_plan_stops_before_second_call(bad):
    model = StructuredFake([bad])
    with pytest.raises(ValueError):
        ground(Image.new('RGB',(1000,800)), 'volume', model, 'panel')
    assert len(model.records) == 1
    assert model.records[0].error is not None


def test_invalid_refinement_is_failed_not_silently_clamped():
    model = StructuredFake([plan(), {'x':9999,'y':0}])
    with pytest.raises(ValueError):
        ground(Image.new('RGB',(1000,800)), 'volume', model, 'panel')
    assert model.records[-1].error is not None


def test_plain_ablation_omits_manual_rules():
    model = StructuredFake([plan(), {'x':100,'y':80}])
    result = ground(Image.new('RGB',(1000,800)), 'volume', model, 'panel_plain')
    assert result.metadata['knowledge_version'] is None
    assert 'Channel Rack rows' not in model.records[0].prompt


@pytest.mark.parametrize('space,point', [('pixels',Point(x=400,y=200)),
                                       ('normalized_1000',Point(x=250,y=250)),
                                       ('normalized_1',Point(x=.25,y=.25))])
def test_explicit_coordinate_conversion(space,point):
    assert to_pixels(point,(1600,800),space) == Point(x=400,y=200)


@pytest.mark.parametrize('x', [-1,1000,float('inf')])
def test_bad_normalized_points_rejected(x):
    with pytest.raises(ValueError):
        to_pixels(Point(x=x,y=0),(1600,800),'normalized_1000')


def test_normalized_box_can_touch_image_boundary():
    assert Box(left=0,top=0,right=1000,bottom=1000).pixels((1600,800),'normalized_1000') == (0,0,1600,800)
