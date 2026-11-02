"""Pinned FL Studio benchmark preparation and explicit, bounded experiment runs."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import os
import shutil
import urllib.parse
import urllib.request

from flground.evaluation import prepare, evaluate, score, read_jsonl
from flground.model import FireworksModel

REVISION = '210e78d3844251110bff86c95835ebd37a6930fa'
BASE = f'https://huggingface.co/datasets/likaixin/ScreenSpot-Pro/resolve/{REVISION}/'
CALLS = {'baseline':1, 'knowledge':1, 'refine':2, 'panel_plain':2, 'panel':2, 'panel_verify':3, 'panel_fast':1}


def download(remote, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix+'.partial')
    with urllib.request.urlopen(BASE+urllib.parse.quote(remote, safe='/'), timeout=120) as response:
        with temporary.open('wb') as stream:
            shutil.copyfileobj(response, stream)
    temporary.replace(target)


def prepare_data(root):
    if (root/'prepared').exists():
        raise ValueError('Dataset is already prepared; use its existing manifest')
    annotation = root/'raw/annotations/fruitloops_windows.json'
    download('annotations/fruitloops_windows.json', annotation)
    rows = json.loads(annotation.read_text())
    if len(rows) != 57:
        raise ValueError('Pinned subset does not contain the expected 57 examples')
    names = sorted({row['img_filename'] for row in rows})
    for index, name in enumerate(names,1):
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Unsafe dataset image path')
        target = root/'raw/images'/name
        if not target.exists():
            download('images/'+name,target)
        print(f'Downloaded screenshot {index}/{len(names)}',flush=True)
    prepare(annotation,root/'raw/images',root/'prepared',REVISION)
    print(json.dumps({'revision':REVISION,'examples':len(rows),'screenshots':len(names),
                      'dev':len(read_jsonl(root/'prepared/dev.jsonl')),
                      'heldout':len(read_jsonl(root/'prepared/heldout.jsonl'))},indent=2))


def run_suite(dataset, output, split, strategies, execute, concurrency=1, max_tokens=None):
    if not 1 <= concurrency <= 20 or (max_tokens is not None and max_tokens <= 0):
        raise ValueError('Invalid concurrency or token budget')
    if not strategies or strategies[0] != 'baseline' or len(set(strategies)) != len(strategies):
        raise ValueError('Specify baseline first, then distinct candidate strategies')
    tasks = dataset/({'full':'tasks.jsonl','dev':'dev.jsonl','heldout':'heldout.jsonl'}[split])
    n = len(read_jsonl(tasks))
    plan = {'split':split,'examples':n,'strategies':strategies,
            'concurrency':concurrency, 'max_tokens_override':max_tokens,
            'maximum_calls':n*sum(CALLS[s] for s in strategies),
            'coordinate_space':'normalized_1000','temperature':0,
            'primary_candidate':strategies[1] if len(strategies)>1 else None,
            'output':str(output),'estimated_cost':None,
            'evaluation_note':'Full-set result is exploratory if any official examples influenced tuning.'}
    print(json.dumps(plan,indent=2),flush=True)
    if not execute:
        print('Dry run only. Add --execute to make billed model calls.')
        return
    output.mkdir(parents=True,exist_ok=False)
    # Explicit experiment settings; identical for baseline and every candidate.
    os.environ['FLGROUND_COORDINATE_SPACE']='normalized_1000'
    os.environ['FLGROUND_TEMPERATURE']='0'
    plan['started_utc']=datetime.now(timezone.utc).isoformat()
    (output/'plan.json').write_text(json.dumps(plan,indent=2))
    if max_tokens is not None:
        os.environ['FLGROUND_MAX_TOKENS']=str(max_tokens)
    baseline=None
    summaries={}
    for strategy in strategies:
        print(f'Running {strategy}: {n} examples',flush=True)
        destination=output/strategy
        evaluate(tasks,destination,FireworksModel(),strategy,
                 concurrency=concurrency, model_factory=FireworksModel,
                 progress=lambda i,n,row: print(
                     f'{strategy} {i}/{n}: {row["status"]}, {row["latency_seconds"]:.1f}s',flush=True))
        predictions=destination/'predictions.jsonl'
        summary=score(predictions,dataset/'labels.jsonl',baseline)
        summaries[strategy]=summary
        (destination/'summary.json').write_text(json.dumps(summary,indent=2))
        (output/'summary.json').write_text(json.dumps(summaries,indent=2))
        if baseline is None:
            baseline=predictions
        print(f'{strategy}: {summary["correct"]}/{summary["total"]} ({summary["accuracy"]:.1%})',flush=True)
    print(f'Saved paired metrics to {output / "summary.json"}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    prep=sub.add_parser('prepare')
    prep.add_argument('--root',type=Path,default=Path('artifacts/benchmark-data'))
    run=sub.add_parser('run')
    run.add_argument('--dataset',type=Path,default=Path('artifacts/benchmark-data/prepared'))
    run.add_argument('--output',type=Path,required=True)
    run.add_argument('--split',choices=['full','dev','heldout'],default='full')
    run.add_argument('--strategies',nargs='+',choices=list(CALLS),default=['baseline','panel'])
    run.add_argument('--concurrency',type=int,default=1)
    run.add_argument('--max-tokens',type=int,default=None)
    run.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if args.action=='prepare':
        prepare_data(args.root)
    else:
        run_suite(args.dataset,args.output,args.split,args.strategies,args.execute,args.concurrency,args.max_tokens)


if __name__=='__main__':
    main()
