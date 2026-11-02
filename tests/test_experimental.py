import json
import pytest
from PIL import Image
from flground.contracts import CallRecord, Point
from flground.harness import ground


class Fake:
    model = 'fake'
    coordinate_space = 'normalized_1000'
    def __init__(self, responses):
        self.responses = iter(responses)
        self.records = []
    def complete(self, images, prompt, stage):
        response = next(self.responses)
        call = CallRecord(raw_response=json.dumps(response), latency_seconds=.1,
                          prompt=prompt, stage=stage, image_sizes=[im.size for im in images])
        self.records.append(call)
        if response in ('length', 'bad'):
            call.finish_reason = 'length' if response == 'length' else 'stop'
            call.error = 'ValueError'
            raise ValueError('failed')
        return call
    def predict(self, image, instruction):
        call = self.complete([image], instruction, 'predict')
        call.point = Point(x=20,y=30)
        return call


def plan(ambiguous=False):
    return dict(visible=True,panel='Mixer',entity='Slot 2',point=dict(x=500,y=500),ambiguous=ambiguous)


def test_recovery_only_on_length_and_preserves_failed_call():
    model=Fake(['length',dict(x=20,y=30)])
    result=ground(Image.new('RGB',(1000,800)), 'target',model,'panel_recover')
    assert result.point==Point(x=20,y=30)
    assert result.model_calls==2 and result.calls[0].finish_reason=='length'
    assert result.metadata['recovery_reason']=='length'


def test_recovery_does_not_retry_other_errors_or_loop():
    for responses, count in [(['bad'],1),(['length','length'],2)]:
        model=Fake(responses)
        with pytest.raises(ValueError):
            ground(Image.new('RGB',(1000,800)),'target',model,'panel_recover')
        assert len(model.records)==count


@pytest.mark.parametrize('strategy,ambiguous,count',[
    ('panel_compact',True,1),('panel_selective',False,1),('panel_selective',True,2)])
def test_selective_budget_and_global_coordinate_mapping(strategy,ambiguous,count):
    model=Fake([plan(ambiguous),dict(x=750,y=250)])
    result=ground(Image.new('RGB',(2000,1000)),'target',model,strategy)
    assert result.model_calls==count
    assert result.point==(Point(x=1500,y=250) if count==2 else Point(x=1000,y=500))
    if count==2:
        assert len(model.records[1].image_sizes)==2
        assert 'ONLY to IMAGE 1' in model.records[1].prompt


def test_compact_rejects_invalid_geometry():
    response=plan();response['point']['x']=1100
    model=Fake([response])
    with pytest.raises(ValueError):
        ground(Image.new('RGB',(1000,800)),'target',model,'panel_compact')
    assert model.records[0].error=='ValueError'
