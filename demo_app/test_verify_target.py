from PIL import Image
from flground.contracts import Point, CallRecord, GroundingResult
from verify_target import verify_target


def test_verification_maps_crop_coordinates_and_records_initial(tmp_path):
    class Model:
        coordinate_space='normalized_1000'
        def complete(self,images,prompt,stage):
            assert images[0].size==(768,768)
            return CallRecord(raw_response='{"x":500,"y":500}',latency_seconds=.1,stage=stage)
    result=GroundingResult(instruction='target',image_width=2000,image_height=1500,
        point=Point(x=1500,y=1000),strategy='panel_recover',model='fake',calls=[],latency_seconds=1)
    answer=verify_target(Image.new('RGB',(2000,1500)),'target',Model(),result,tmp_path)
    assert answer.point==Point(x=1500,y=1000)
    assert answer.metadata['demo_context_verification']['bounds']==(1116,616,1884,1384)
    assert len(answer.calls)==1
