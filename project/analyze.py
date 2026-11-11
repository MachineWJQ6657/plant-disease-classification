"""Recompute all evaluation tables and confusion matrices from saved predictions."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from prepare import LABELS

def analyze(out):
    p=pd.read_csv(out/'predictions.csv')
    manifest=pd.read_csv(out/'manifest.csv')
    expected_groups={(c,s) for c in ('ordinary','strong') for s in ('pv_test','pd_test')}
    if set(map(tuple,p[['condition','split']].drop_duplicates().to_numpy())) != expected_groups:
        raise ValueError('Expected both conditions on both test sets; refusing incomplete analysis')
    if p.duplicated(['condition','path']).any():
        raise ValueError('Duplicate predictions')
    probabilities=p[[f'p_{k}' for k in range(9)]].to_numpy()
    if not (np.isfinite(probabilities).all() and (probabilities>=0).all() and (probabilities<=1).all()
            and np.allclose(probabilities.sum(1),1,atol=2e-6,rtol=0)
            and np.array_equal(probabilities.argmax(1),p.y_pred.to_numpy())):
        raise ValueError('Invalid probability vectors or predicted labels')
    counts=pd.crosstab(manifest.category,manifest.split)
    counts.to_csv(out/'split_counts.csv')
    ids=list(range(len(LABELS)))
    overall=[]
    classes=[]
    for (cond,split),g in p.groupby(['condition','split']):
        truth,pred=g.y_true,g.y_pred
        expected=manifest[manifest.split==split].set_index('path').label.sort_index()
        observed=g.set_index('path').y_true.sort_index()
        assert observed.index.equals(expected.index) and np.array_equal(observed,expected)
        acc=accuracy_score(truth,pred)
        f=f1_score(truth,pred,labels=ids,average=None,zero_division=0)
        rec=recall_score(truth,pred,labels=ids,average=None,zero_division=0)
        overall.append(dict(condition=cond,split=split,n=len(g),correct=int((truth==pred).sum()),accuracy=acc,macro_f1=f.mean()))
        for k,(label,_,_) in enumerate(LABELS):
            classes.append(dict(condition=cond,split=split,category=label,support=int((truth==k).sum()),recall=rec[k],f1=f[k]))
        names=[v[0] for v in LABELS]
        pd.DataFrame(confusion_matrix(truth,pred,labels=ids),index=names,columns=names).rename_axis('true_label').to_csv(out/f'confusion_{cond}_{split}.csv')
    overall=pd.DataFrame(overall)
    overall.to_csv(out/'overall.csv',index=False)
    c=pd.DataFrame(classes)
    c.to_csv(out/'per_class.csv',index=False)
    wide=c.pivot(index=['category','condition'],columns='split',values=['recall','f1'])
    for m in ['recall','f1']:
        wide[(m,'gap_pp')]=100*(wide[(m,'pv_test')]-wide[(m,'pd_test')])
    wide.columns=['_'.join(x) for x in wide.columns]
    wide.to_csv(out/'gaps.csv')
    # Resample paired predictions within true class so rare classes keep their observed support.
    g=p[p.split=='pd_test'].pivot(index=['path','y_true'],columns='condition',values='y_pred').reset_index()
    truth=g.y_true.to_numpy()
    a,b=g.ordinary.to_numpy(),g.strong.to_numpy()
    rng=np.random.default_rng(664)
    groups=[np.flatnonzero(truth==k) for k in ids]
    boot=[]
    for _ in range(2000):
        ix=np.concatenate([rng.choice(idx,len(idx),replace=True) for idx in groups])
        y=truth[ix]
        def scores(pred):
            cm=np.bincount(9*y+pred[ix],minlength=81).reshape(9,9)
            return np.array([np.trace(cm)/len(y),np.mean(2*cm.diagonal()/(cm.sum(0)+cm.sum(1)))])
        boot.append(scores(b)-scores(a))
    intervals=np.quantile(boot,[.025,.975],axis=0)
    pd.DataFrame(intervals,index=['p025','p975'],columns=['accuracy_delta','macro_f1_delta']).to_csv(out/'paired_field_bootstrap.csv')
    print(overall.to_string(index=False))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,default=Path('results'))
    analyze(p.parse_args().out)
