"""Audio goes only to local Gemma; no screenshots or Fireworks credentials here."""
import base64
from pathlib import Path
from urllib.parse import urlsplit
from openai import OpenAI


def transcribe(path, endpoint):
    url=urlsplit(endpoint)
    if url.scheme!='http' or url.hostname not in ('localhost','127.0.0.1','::1') or url.username or url.password:
        raise ValueError('Transcription requires a loopback HTTP endpoint')
    with OpenAI(base_url=endpoint,api_key='local',timeout=90,max_retries=0) as client:
        models=client.models.list().data
        if len(models)!=1: raise ValueError('Expected one local Gemma model')
        response=client.chat.completions.create(model=models[0].id,temperature=0,max_tokens=256,
            messages=[{'role':'user','content':[
                {'type':'text','text':'Transcribe the spoken words exactly. Return only the transcript. Do not answer the question. For silence return an empty string.'},
                {'type':'input_audio','input_audio':{'data':base64.b64encode(Path(path).read_bytes()).decode(),'format':'wav'}}]}])
        if response.choices[0].finish_reason!='stop': raise ValueError('Incomplete audio transcription')
        return (response.choices[0].message.content or '').strip()
