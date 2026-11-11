"""Replay shipped checkpoints against raw test images; optionally compare saved scores."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from experiment import model, encode, predict
from prepare import LABELS, write_csv
from paths import disk_path


def configure_replay(protocol):
    """Restore the numerical settings recorded for the independently trained models."""
    if protocol=='fresh_views_v1':
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    return dict(protocol=protocol or 'legacy',
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32)


def evaluate(data, results, out, device, compare=False):
    torch.set_num_threads(4)
    metadata=torch.load(results/'ordinary.pt',map_location='cpu',weights_only=True)
    expected_protocol=metadata.get('protocol')
    numerical_settings=configure_replay(expected_protocol)
    del metadata
    with (results/'manifest.csv').open(encoding='utf-8',newline='') as f:
        rows=list(csv.DictReader(f))
    tests=[r for r in rows if r['split'] in {'pv_test','pd_test'}]
    for r in tests:
        path=disk_path(data,r['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=r['sha256']:
            raise ValueError(f'Raw image hash mismatch: {path}')
    predictions=[]
    for condition in ['ordinary','strong']:
        ckpt=torch.load(results/f'{condition}.pt',map_location=device,weights_only=True)
        if ckpt.get('protocol')!=expected_protocol:
            raise ValueError('Checkpoint pair uses different numerical protocols')
        if ckpt['labels'] != [r[0] for r in LABELS] or ckpt['size'] != 128 or ckpt['condition'] != condition:
            raise ValueError('Checkpoint metadata mismatch')
        net=model(False).to(device=device,memory_format=torch.channels_last)
        net.load_state_dict(ckpt['model'])
        net.eval()
        for split in ['pv_test','pd_test']:
            subset=[r for r in tests if r['split']==split]
            feats=encode(net.features[:-2],subset,data,'ordinary',False,0,device)
            probs=predict(net,feats,device).numpy()
            for r,prob in zip(subset,probs):
                predictions.append(dict(path=r['path'],split=split,condition=condition,y_true=r['label'],
                    y_pred=int(prob.argmax()),confidence=float(prob.max()),**{f'p_{k}':float(p) for k,p in enumerate(prob)}))
            print(condition,split,len(subset),'scored',flush=True)
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'predictions.csv',predictions)
    report=dict(status='PASS',raw_image_hashes_checked=len(tests),prediction_rows=len(predictions),device=device,
                torch=torch.__version__,numerical_settings=numerical_settings)
    if compare:
        with (results/'predictions.csv').open(encoding='utf-8',newline='') as f:
            saved={(r['condition'],r['path']):r for r in csv.DictReader(f)}
        if set(saved)!={(r['condition'],r['path']) for r in predictions}:
            raise ValueError('Prediction keys differ')
        mismatches=0
        max_delta=0.
        for r in predictions:
            old=saved[r['condition'],r['path']]
            mismatches+=int(r['y_pred']!=int(old['y_pred']))
            max_delta=max(max_delta,max(abs(r[f'p_{k}']-float(old[f'p_{k}'])) for k in range(9)))
        report.update(top1_mismatches=mismatches,max_probability_delta=max_delta)
        report['status']='PASS' if mismatches==0 and max_delta<.005 else 'REVIEW'
    (out/'replay_verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
    if report['status']!='PASS':
        raise SystemExit('Replay differs materially. Review hardware/library variation before reporting reproduction.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('data'))
    p.add_argument('--results',type=Path,default=Path('results'))
    p.add_argument('--out',type=Path,default=Path('replay'))
    p.add_argument('--device',default='cpu')
    p.add_argument('--compare',action='store_true')
    a=p.parse_args()
    if a.out.resolve()==a.results.resolve():
        p.error('--out must differ from --results to preserve the original evidence')
    evaluate(a.data,a.results,a.out,a.device,a.compare)
