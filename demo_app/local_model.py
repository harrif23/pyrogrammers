"""Demo-only adapter. Shares the existing request tracing/parser, never reads .env."""
from urllib.parse import urlsplit
from openai import OpenAI
from flground.model import FireworksModel


class LocalModel(FireworksModel):
    def __init__(self, base_url='http://127.0.0.1:8080/v1', coordinate_space='normalized_1000'):
        url=urlsplit(base_url)
        if url.scheme != 'http' or url.hostname not in ('127.0.0.1','localhost','::1') or url.username or url.password:
            raise ValueError('The demo local endpoint must be a loopback HTTP URL')
        if coordinate_space not in ('pixels','normalized_1000','normalized_1'):
            raise ValueError('Invalid coordinate space')
        self.client=OpenAI(base_url=base_url,api_key='local',timeout=90,max_retries=0)
        models=self.client.models.list().data
        if len(models)!=1:
            raise ValueError('Expected one loaded model in llama-server')
        self.model=models[0].id
        self.coordinate_space=coordinate_space
        self.max_tokens=2048
        self.temperature=0
        self.records=[]

    def manifest(self):
        data=super().manifest()
        data.update(provider='local llama-server',timeout_seconds=90)
        return data

    def request_options(self, stage):
        if stage == 'panel_plan':
            from flground.panel import PanelPlan
            return {'response_format': {'type': 'json_schema', 'json_schema': {
                'name': 'panel_plan', 'strict': True, 'schema': PanelPlan.model_json_schema()}}}
        return {}
