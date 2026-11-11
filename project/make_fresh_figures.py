"""Create report figures and LaTeX tables for the independent three-seed training."""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prepare import LABELS


def build(root,dest):
    dest.mkdir(parents=True,exist_ok=True)
    summary=pd.read_csv(root/'summary.csv').set_index(['condition','split'])
    per_seed=pd.read_csv(root/'per_seed_metrics.csv')
    classes=pd.read_csv(root/'class_summary.csv')
    history=pd.read_csv(root/'all_history.csv')
    effects=pd.read_csv(root/'paired_seed_summary.csv').set_index('split')
    deltas=pd.read_csv(root/'paired_seed_deltas.csv')
    audit=pd.read_csv(root/'training_audit.csv')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(9.5,3.3),layout='constrained')
    for ax,metric,title in zip(axes,['accuracy','macro_f1'],['Accuracy','Macro-F1']):
        for j,(condition,color) in enumerate([('ordinary','#555555'),('strong','#BBBBBB')]):
            means=[100*summary.loc[(condition,s),metric+'_mean'] for s in ['pv_test','pd_test']]
            stds=[100*summary.loc[(condition,s),metric+'_std'] for s in ['pv_test','pd_test']]
            positions=np.arange(2)+(j-.5)*.32
            bars=ax.bar(positions,means,.32,color=color,edgecolor='black',linewidth=.5,
                yerr=stds,capsize=4,label='Ordinary' if condition=='ordinary' else 'Crop + color')
            ax.bar_label(bars,fmt='%.1f',padding=7,fontsize=8)
            for pos,split in zip(positions,['pv_test','pd_test']):
                values=100*per_seed.loc[(per_seed.condition==condition)&(per_seed.split==split)].sort_values('seed')[metric].to_numpy()
                ax.scatter(pos+np.array([-.045,0,.045]),values,s=15,c='black',zorder=3)
        ax.set(xticks=[0,1],xticklabels=['PlantVillage','PlantDoc'],ylim=(0,111),ylabel=title+' (%)')
        ax.set_yticks([0,25,50,75,100])
    axes[0].legend(frameon=False,fontsize=8,loc='upper right')
    for ext in ['pdf','png']:fig.savefig(dest/f'fresh_performance.{ext}',dpi=180)
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(9.5,3),layout='constrained')
    for condition,color in [('ordinary','#333333'),('strong','#999999')]:
        h=history[history.condition==condition]
        for ax,metric,label in zip(axes,['train_loss','val_supported_macro_f1'],['Training cross-entropy','Validation macro-F1 (%)']):
            scale=100 if metric=='val_supported_macro_f1' else 1
            grouped=h.groupby('epoch')[metric].agg(['mean','std'])
            x=grouped.index.to_numpy();mean=scale*grouped['mean'].to_numpy();std=scale*grouped['std'].to_numpy()
            ax.plot(x,mean,'o-' if condition=='ordinary' else 's--',color=color,label=condition)
            ax.fill_between(x,mean-std,mean+std,color=color,alpha=.12)
            ax.set(xlabel='Epoch',ylabel=label,xticks=range(1,9))
    axes[0].legend(frameon=False)
    for ext in ['pdf','png']:fig.savefig(dest/f'fresh_history.{ext}',dpi=180)
    plt.close(fig)

    names=['Healthy','Early blight','Bacterial spot','Late blight','Mosaic','Yellow curl','Leaf mold','Septoria','Spider mites']
    fig,axes=plt.subplots(1,2,figsize=(10,4.8),layout='constrained')
    for ax,condition in zip(axes,['ordinary','strong']):
        cm=pd.read_csv(root/'seed_664'/f'confusion_{condition}_pd_test.csv',index_col=0).to_numpy()
        row=cm/cm.sum(1,keepdims=True)
        ax.imshow(row,vmin=0,vmax=1,cmap='Greys')
        for i in range(9):
            for j in range(9):
                ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=7,color='white' if row[i,j]>.5 else 'black')
        ax.set(xticks=range(9),yticks=range(9),xticklabels=names,yticklabels=names,title=condition,
            xlabel='Predicted class',ylabel='True class')
        plt.setp(ax.get_xticklabels(),rotation=55,ha='right',fontsize=8)
        plt.setp(ax.get_yticklabels(),fontsize=8)
    for ext in ['pdf','png']:fig.savefig(dest/f'fresh_confusions.{ext}',dpi=180)
    plt.close(fig)

    def row(values):return ' & '.join(str(v) for v in values)+r' \\'
    table=[]
    for condition in ['ordinary','strong']:
        values=['Ordinary' if condition=='ordinary' else 'Crop + color']
        for split,metric in [('pv_test','accuracy'),('pd_test','accuracy'),('pv_test','macro_f1'),('pd_test','macro_f1')]:
            s=summary.loc[(condition,split)]
            values.append(f"${100*s[metric+'_mean']:.2f} \\pm {100*s[metric+'_std']:.2f}$")
        table.append(row(values))
    (dest/'fresh_summary.tex').write_text('\n'.join(table)+'\n',encoding='utf-8')
    table=[]
    for seed in [664,665,666]:
        for condition in ['ordinary','strong']:
            m=per_seed[(per_seed.seed==seed)&(per_seed.condition==condition)].set_index('split')
            epoch=int(audit[(audit.seed==seed)&(audit.condition==condition)].selected_epoch.iloc[0])
            table.append(row([seed,'Ordinary' if condition=='ordinary' else 'Crop + color',epoch]+[
                f'{100*m.loc[s,metric]:.2f}' for s,metric in [('pv_test','accuracy'),('pd_test','accuracy'),('pv_test','macro_f1'),('pd_test','macro_f1')]]))
    (dest/'fresh_seeds.tex').write_text('\n'.join(table)+'\n',encoding='utf-8')
    field_delta=deltas[deltas.split=='pd_test'].sort_values('seed')
    (dest/'fresh_deltas.tex').write_text('\n'.join(row([int(r.seed),f'{r.accuracy_delta_pp:+.2f}',f'{r.macro_f1_delta_pp:+.2f}']) for r in field_delta.itertuples())+'\n',encoding='utf-8')
    table=[]
    class_index=classes.set_index(['condition','split','category'])
    for name,_,_ in LABELS:
        values=[name,int(class_index.loc[('ordinary','pd_test',name),'support'])]
        for condition in ['ordinary','strong']:
            clean=100*class_index.loc[(condition,'pv_test',name),'recall_mean']
            field=100*class_index.loc[(condition,'pd_test',name),'recall_mean']
            values += [f'{clean:.1f}',f'{field:.1f}',f'{clean-field:.1f}']
        table.append(row(values))
    (dest/'fresh_recall.tex').write_text('\n'.join(table)+'\n',encoding='utf-8')

    macros={}
    for condition,prefix in [('ordinary','Ord'),('strong','Aug')]:
        for split,domain in [('pv_test','Clean'),('pd_test','Field')]:
            for metric,label in [('accuracy','Acc'),('macro_f1','Fone')]:
                s=summary.loc[(condition,split)]
                macros[f'Fresh{prefix}{domain}{label}']=f"{100*s[metric+'_mean']:.2f}"
                macros[f'Fresh{prefix}{domain}{label}SD']=f"{100*s[metric+'_std']:.2f}"
    for metric,label in [('accuracy','Acc'),('macro_f1','Fone')]:
        e=effects.loc['pd_test']
        macros[f'FreshDelta{label}']=f"{e[metric+'_delta_mean_pp']:+.2f}"
        macros[f'FreshDelta{label}SD']=f"{e[metric+'_delta_std_pp']:.2f}"
    (dest/'fresh_numbers.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in macros.items())+'\n',encoding='utf-8')

    confidence=[];predicted=[]
    for seed in [664,665,666]:
        p=pd.read_csv(root/f'seed_{seed}'/'predictions.csv')
        for condition in ['ordinary','strong']:
            field=p[(p.condition==condition)&(p.split=='pd_test')]
            high=field[field.confidence>=.9]
            confidence.append(dict(seed=seed,condition=condition,n=len(field),high_confidence_n=len(high),
                high_confidence_correct=int((high.y_true==high.y_pred).sum()),
                high_confidence_accuracy=float((high.y_true==high.y_pred).mean())))
            for k,(name,_,_) in enumerate(LABELS):
                predicted.append(dict(seed=seed,condition=condition,category=name,true_count=int((field.y_true==k).sum()),
                    predicted_count=int((field.y_pred==k).sum())))
    pd.DataFrame(confidence).to_csv(root/'confidence_diagnostics.csv',index=False)
    pd.DataFrame(predicted).to_csv(root/'predicted_class_counts.csv',index=False)
    confidence=pd.DataFrame(confidence)
    table=[]
    for r in confidence.itertuples():
        table.append(row([r.seed,'Ordinary' if r.condition=='ordinary' else 'Crop + color',
            r.high_confidence_n,r.high_confidence_correct,f'{100*r.high_confidence_accuracy:.2f}']))
    (dest/'fresh_confidence.tex').write_text('\n'.join(table)+'\n',encoding='utf-8')
    predicted=pd.DataFrame(predicted)
    paragraphs=[]
    for condition,label in [('ordinary','Ordinary'),('strong','Strong')]:
        totals=predicted[predicted.condition==condition].groupby('category').agg(
            mean=('predicted_count','mean'),sd=('predicted_count','std'),true=('true_count','first'))
        name=totals['mean'].idxmax();r=totals.loc[name]
        paragraphs.append(f"{label} training most often predicted {name.lower()}, assigning that label to "
            f"{r['mean']:.1f} of 742 field images on average (SD {r['sd']:.1f}); its true support was {int(r['true'])}.")
    field_classes=classes[classes.split=='pd_test'].pivot(index='category',columns='condition',values='recall_mean')
    change=100*(field_classes['strong']-field_classes['ordinary'])
    paragraphs.append(f"The largest signed mean class-recall change was {change.max():+.1f} pp for {change.idxmax().lower()}, "
        f"and the smallest was {change.min():+.1f} pp for {change.idxmin().lower()}.")
    (dest/'fresh_errors.tex').write_text(' '.join(paragraphs)+'\n',encoding='utf-8')
    if (field_delta.accuracy_delta_pp>0).all() and (field_delta.macro_f1_delta_pp>0).all():
        conclusion='Strong augmentation improved both field metrics in all three fitted seed pairs, while a substantial clean-to-field gap remained.'
    elif effects.loc['pd_test','accuracy_delta_mean_pp']>0 and effects.loc['pd_test','macro_f1_delta_mean_pp']>0:
        conclusion='Strong augmentation improved both field metrics on average, but the paired improvements were not consistent across all three seeds.'
    else:
        conclusion='Strong augmentation did not improve both average field metrics under the fresh-view protocol, and the seed-wise results limit any general claim of benefit.'
    (dest/'fresh_interpretation.tex').write_text('\\newcommand{\\FreshConclusion}{'+conclusion+'}\n',encoding='utf-8')
    demo=pd.read_csv(root/'seed_664'/'predictions.csv')
    credits=pd.read_csv(root.parent.parent/'examples/credits.csv')
    demo_macros={}
    for filename,suffix in [('field.jpg','One'),('field_second.jpg','Two')]:
        path=credits.loc[credits['use']==filename,'path'].iloc[0]
        for condition,prefix in [('ordinary','Ord'),('strong','Aug')]:
            selected=demo[(demo.path==path)&(demo.condition==condition)].iloc[0]
            demo_macros[f'Demo{prefix}{suffix}']=LABELS[int(selected.y_pred)][0].lower()
    (dest/'fresh_demo.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in demo_macros.items())+'\n',encoding='utf-8')
    (root/'interpretation.json').write_text(json.dumps(dict(conclusion=conclusion,field_error_summary=paragraphs),indent=2)+'\n',encoding='utf-8')
    print(summary.to_string())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=Path('results/fresh_views'))
    p.add_argument('--dest',type=Path,default=Path('report/figures'))
    a=p.parse_args();build(a.results,a.dest)
