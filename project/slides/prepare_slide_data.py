"""Refresh simple presentation tables from the fresh three-seed experiment."""
from pathlib import Path
import json
import pandas as pd
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prepare import LABELS
project=Path(__file__).resolve().parents[1]
root=project/'results/fresh_views'
summary=pd.read_csv(root/'summary.csv').set_index(['condition','split'])
metrics=pd.read_csv(root/'per_seed_metrics.csv').set_index(['seed','condition','split'])
audit=pd.read_csv(root/'training_audit.csv').set_index(['seed','condition'])
effects=pd.read_csv(root/'paired_seed_summary.csv').set_index('split')
deltas=pd.read_csv(root/'paired_seed_deltas.csv')
classes=pd.read_csv(root/'class_summary.csv').set_index(['condition','split','category'])
rows=[]
for split,metric,title in [('pv_test','accuracy','Clean accuracy'),('pd_test','accuracy','Field accuracy'),('pv_test','macro_f1','Clean macro-F1'),('pd_test','macro_f1','Field macro-F1')]:
    rows.append([title]+[f"{100*summary.loc[(c,split),metric+'_mean']:.2f} ± {100*summary.loc[(c,split),metric+'_std']:.2f}" for c in ['ordinary','strong']])
seeds=[]
for seed in [664,665,666]:
    for condition in ['ordinary','strong']:
        r=metrics.loc[(seed,condition,'pd_test')]
        seeds.append([str(seed),'Ordinary' if condition=='ordinary' else 'Crop + color',str(int(audit.loc[(seed,condition),'selected_epoch'])),f'{100*r.accuracy:.2f}',f'{100*r.macro_f1:.2f}'])
paired=[[str(int(r.seed)),f'{r.accuracy_delta_pp:+.2f}',f'{r.macro_f1_delta_pp:+.2f}'] for r in deltas[deltas.split=='pd_test'].sort_values('seed').itertuples()]
e=effects.loc['pd_test']
paired.append(['Mean ± SD',f'{e.accuracy_delta_mean_pp:+.2f} ± {e.accuracy_delta_std_pp:.2f}',f'{e.macro_f1_delta_mean_pp:+.2f} ± {e.macro_f1_delta_std_pp:.2f}'])
recall=[]
for name,_,_ in LABELS:
    ordinary=100*classes.loc[('ordinary','pd_test',name),'recall_mean']
    strong=100*classes.loc[('strong','pd_test',name),'recall_mean']
    recall.append([name,str(int(classes.loc[('ordinary','pd_test',name),'support'])),f'{ordinary:.1f}',f'{strong:.1f}',f'{strong-ordinary:+.1f}'])
credits=pd.read_csv(project/'examples/credits.csv');p=pd.read_csv(root/'seed_664/predictions.csv')
examples=[]
for filename in ['field.jpg','field_second.jpg']:
    path=credits.loc[credits['use']==filename,'path'].iloc[0]
    record={'filename':filename}
    for c in ['ordinary','strong']:
        r=p[(p.path==path)&(p.condition==c)].iloc[0]
        record[c]=LABELS[int(r.y_pred)][0]
    examples.append(record)
interpretation=json.loads((root/'interpretation.json').read_text())
(project/'slides/slide_data.json').write_text(json.dumps(dict(overall=rows,seeds=seeds,paired=paired,recall=recall,examples=examples,interpretation=interpretation),indent=2)+'\n',encoding='utf-8')
