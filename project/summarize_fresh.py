"""Aggregate the three prespecified seed pairs and audit their training budgets."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from verify import audit


def summarize(root):
    protocol=json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    completion=json.loads((root/'run_complete.json').read_text(encoding='utf-8'))
    lock=json.loads((root/'selection_locked.json').read_text(encoding='utf-8'))
    assert completion['status']=='PASS'
    metrics=[];classes=[];histories=[];checks=[]
    for seed in protocol['seeds']:
        folder=root/f'seed_{seed}'
        result=audit(folder)
        assert result['status']=='PASS'
        history=pd.read_csv(folder/'history.csv')
        histories.append(history.assign(seed=seed))
        a=history[history.condition=='ordinary'].sort_values('epoch')
        b=history[history.condition=='strong'].sort_values('epoch')
        assert a.epoch.tolist()==b.epoch.tolist()==list(range(1,9))
        assert a.sample_order_sha256.tolist()==b.sample_order_sha256.tolist()
        assert a.optimizer_steps.eq(126).all() and b.optimizer_steps.eq(126).all()
        records=[]
        for condition in ['ordinary','strong']:
            done=json.loads((folder/f'{condition}_training_complete.json').read_text(encoding='utf-8'))
            assert done['optimizer_steps']==1008 and done['image_presentations']==96288
            assert done['frozen_prefix_unchanged']
            digest=hashlib.sha256((folder/f'{condition}.pt').read_bytes()).hexdigest()
            assert digest==lock['checkpoint_sha256'][f'{seed}/{condition}']
            records.append(done)
            checks.append(dict(seed=seed,condition=condition,optimizer_steps=done['optimizer_steps'],
                image_presentations=done['image_presentations'],selected_epoch=result['selected_epochs'][condition],
                frozen_prefix_unchanged=True,checkpoint_sha256=digest))
        assert records[0]['initial_state_sha256']==records[1]['initial_state_sha256']
        metrics.append(pd.read_csv(folder/'overall.csv').assign(seed=seed))
        classes.append(pd.read_csv(folder/'per_class.csv').assign(seed=seed))
    m=pd.concat(metrics,ignore_index=True)
    c=pd.concat(classes,ignore_index=True)
    h=pd.concat(histories,ignore_index=True)
    m.to_csv(root/'per_seed_metrics.csv',index=False)
    c.to_csv(root/'per_seed_class_metrics.csv',index=False)
    h.to_csv(root/'all_history.csv',index=False)
    summary=m.groupby(['condition','split']).agg(seeds=('seed','nunique'),
        accuracy_mean=('accuracy','mean'),accuracy_std=('accuracy','std'),
        macro_f1_mean=('macro_f1','mean'),macro_f1_std=('macro_f1','std')).reset_index()
    summary.to_csv(root/'summary.csv',index=False)
    paired=[]
    for seed in protocol['seeds']:
        for split in ['pv_test','pd_test']:
            sub=m[(m.seed==seed)&(m.split==split)].set_index('condition')
            paired.append(dict(seed=seed,split=split,
                accuracy_delta_pp=100*(sub.loc['strong','accuracy']-sub.loc['ordinary','accuracy']),
                macro_f1_delta_pp=100*(sub.loc['strong','macro_f1']-sub.loc['ordinary','macro_f1'])))
    paired=pd.DataFrame(paired)
    paired.to_csv(root/'paired_seed_deltas.csv',index=False)
    effects=paired.groupby('split').agg(seeds=('seed','nunique'),accuracy_delta_mean_pp=('accuracy_delta_pp','mean'),
        accuracy_delta_std_pp=('accuracy_delta_pp','std'),macro_f1_delta_mean_pp=('macro_f1_delta_pp','mean'),
        macro_f1_delta_std_pp=('macro_f1_delta_pp','std')).reset_index()
    effects.to_csv(root/'paired_seed_summary.csv',index=False)
    wide=c.pivot(index=['seed','condition','category'],columns='split',values=['recall','f1'])
    for metric in ['recall','f1']:
        wide[(metric,'gap_pp')]=100*(wide[(metric,'pv_test')]-wide[(metric,'pd_test')])
    wide.columns=['_'.join(x) for x in wide.columns]
    wide.reset_index().to_csv(root/'per_seed_class_gaps.csv',index=False)
    class_summary=c.groupby(['condition','split','category']).agg(seeds=('seed','nunique'),support=('support','first'),
        recall_mean=('recall','mean'),recall_std=('recall','std'),f1_mean=('f1','mean'),f1_std=('f1','std')).reset_index()
    class_summary.to_csv(root/'class_summary.csv',index=False)
    audits=pd.DataFrame(checks);audits.to_csv(root/'training_audit.csv',index=False)
    events=[json.loads(line) for line in (root/'events.jsonl').read_text(encoding='utf-8').splitlines()]
    epochs=[e for e in events if e['event']=='epoch_complete']
    assert len({(e['seed'],e['condition'],e['epoch']) for e in epochs})==48
    locked_index=next(i for i,e in enumerate(events) if e['event']=='all_checkpoints_locked_before_test_evaluation')
    assert all(i>locked_index for i,e in enumerate(events) if e['event']=='test_predictions_saved')
    record=dict(status='PASS',seeds=protocol['seeds'],trained_models=6,unique_completed_epochs=48,
        optimizer_steps=6048,total_image_presentations=577728,
        identical_initialization_within_seed=True,identical_sample_order_within_seed=True,
        all_six_checkpoints_locked_before_test_predictions=True,original_split_preserved=True,
        all_per_seed_artifact_audits_passed=True,
        interpretation='Mean and sample SD describe three fitted seeds on the same test images, not three independent test datasets.')
    (root/'aggregate_verification.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(summary.to_string(index=False))
    print(paired.to_string(index=False))
    print(json.dumps(record,indent=2))
    return summary,effects


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=Path('results/fresh_views'))
    summarize(p.parse_args().out)
