"""Training-only constant baseline and checkpoint/shift diagnostics."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from experiment import model
from prepare import LABELS

def diagnose(out):
    manifest=pd.read_csv(out/'manifest.csv')
    p=pd.read_csv(out/'predictions.csv')
    history=pd.read_csv(out/'history.csv')
    majority=int(manifest.loc[manifest.split=='pv_train','label'].value_counts().idxmax())
    baseline=[]
    for split in ['pv_test','pd_test']:
        y=manifest.loc[manifest.split==split,'label'].to_numpy()
        z=np.full(len(y),majority)
        baseline.append(dict(method='source_training_majority',class_index=majority,category=LABELS[majority][0],split=split,n=len(y),accuracy=accuracy_score(y,z),macro_f1=f1_score(y,z,labels=range(9),average='macro',zero_division=0)))
    pd.DataFrame(baseline).to_csv(out/'majority_baseline.csv',index=False)
    counts=[]
    diagnostic=[]
    for (condition,split),g in p.groupby(['condition','split']):
        for k,(name,_,_) in enumerate(LABELS):
            counts.append(dict(condition=condition,split=split,category=name,true_count=int((g.y_true==k).sum()),predicted_count=int((g.y_pred==k).sum())))
        high=g.confidence>=.9
        diagnostic.append(dict(condition=condition,split=split,n=len(g),mean_confidence=float(g.confidence.mean()),high_confidence_n=int(high.sum()),high_confidence_accuracy=float((g.loc[high,'y_true']==g.loc[high,'y_pred']).mean())))
    pd.DataFrame(counts).to_csv(out/'predicted_class_counts.csv',index=False)
    pd.DataFrame(diagnostic).to_csv(out/'confidence_diagnostics.csv',index=False)

    # This check fetches the official ImageNet weights once if not already cached.
    torch.set_num_threads(4)
    initial=model(True)
    reference=initial.features[:-2].state_dict()
    reports=[]
    for condition in ['ordinary','strong']:
        checkpoint=torch.load(out/f'{condition}.pt',map_location='cpu',weights_only=True)
        net=model(False);net.load_state_dict(checkpoint['model'])
        net.features[:-2].requires_grad_(False)
        frozen=net.features[:-2].state_dict()
        unchanged=all(torch.equal(reference[k],frozen[k]) for k in reference)
        h=history[history.condition==condition]
        epoch=int(h.loc[h.val_supported_macro_f1.idxmax(),'epoch'])
        report=dict(condition=condition,selected_epoch=checkpoint['epoch'],history_best_epoch=epoch,
            total_parameters=sum(x.numel() for x in net.parameters()),trainable_parameters=sum(x.numel() for x in net.parameters() if x.requires_grad),
            class_order_matches=checkpoint['labels']==[x[0] for x in LABELS],input_size=checkpoint['size'],
            prefix_exactly_matches_official_imagenet=unchanged)
        if not (unchanged and report['class_order_matches'] and epoch==checkpoint['epoch'] and checkpoint['size']==128):
            raise ValueError(f'Checkpoint audit failed: {report}')
        reports.append(report)
    result=dict(status='PASS',note='Exact official ImageNet prefix comparison verifies the frozen-prefix claim; no retraining is implied.',checkpoints=reports)
    (out/'checkpoint_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(pd.DataFrame(baseline).to_string(index=False))
    print(pd.DataFrame(diagnostic).to_string(index=False))
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('results'))
    diagnose(parser.parse_args().out)
