"""Generate publication figures and LaTeX tables from saved CSV evidence."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prepare import LABELS

def figures(results, dest):
    dest.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    overall=pd.read_csv(results/'overall.csv').set_index(['condition','split'])
    fig,axs=plt.subplots(1,2,figsize=(9,3.2),layout='constrained')
    for ax,metric,title in zip(axs,['accuracy','macro_f1'],['Accuracy','Macro-F1']):
        x=np.arange(2)
        for j,(cond,color) in enumerate([('ordinary','#555555'),('strong','#BBBBBB')]):
            values=[100*overall.loc[(cond,s),metric] for s in ['pv_test','pd_test']]
            bars=ax.bar(x+(j-.5)*.32,values,.32,label='Ordinary' if j==0 else 'Crop + color',color=color,edgecolor='black',linewidth=.5)
            ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=9)
        ax.set(xticks=x,xticklabels=['PlantVillage','PlantDoc'],ylim=(0,110),ylabel=f'{title} (%)')
        ax.set_yticks([0,25,50,75,100])
    axs[0].legend(loc='upper right',frameon=False,fontsize=8)
    for ext in ['pdf','png']:fig.savefig(dest/f'performance.{ext}',dpi=180)
    plt.close(fig)

    names=['Healthy','Early blight','Bacterial spot','Late blight','Mosaic','Yellow curl','Leaf mold','Septoria','Spider mites']
    fig,axs=plt.subplots(1,2,figsize=(10,4.8),layout='constrained')
    for ax,cond in zip(axs,['ordinary','strong']):
        cm=pd.read_csv(results/f'confusion_{cond}_pd_test.csv',index_col=0).to_numpy()
        ax.imshow(cm/cm.sum(1,keepdims=True),vmin=0,vmax=1,cmap='Greys')
        for r in range(9):
            for c in range(9):
                ax.text(c,r,str(cm[r,c]),ha='center',va='center',fontsize=7,color='white' if cm[r,c]/cm[r].sum()>.5 else 'black')
        ax.set(xticks=range(9),yticks=range(9),xticklabels=names,yticklabels=names,title='Ordinary' if cond=='ordinary' else 'Crop + color',xlabel='Predicted class',ylabel='True class')
        plt.setp(ax.get_xticklabels(),rotation=55,ha='right',fontsize=8)
        plt.setp(ax.get_yticklabels(),fontsize=8)
    for ext in ['pdf','png']:fig.savefig(dest/f'field_confusions.{ext}',dpi=180)
    plt.close(fig)

    hist=pd.read_csv(results/'history.csv')
    fig,ax=plt.subplots(figsize=(7,2.5),layout='constrained')
    for cond,style in [('ordinary','o-'),('strong','s--')]:
        h=hist[hist.condition==cond]
        ax.plot(h.epoch,100*h.val_supported_macro_f1,style,color='black' if cond=='ordinary' else '#888888',label='Ordinary' if cond=='ordinary' else 'Crop + color')
        row=h.loc[h.val_supported_macro_f1.idxmax()]
        ax.scatter([row.epoch],[100*row.val_supported_macro_f1],s=100,facecolors='none',edgecolors='black')
    ax.set(xlabel='Epoch',ylabel='Validation macro-F1 (%)',xticks=range(1,9),ylim=(74,100))
    ax.legend(frameon=False,loc='lower right')
    for ext in ['pdf','png']:fig.savefig(dest/f'validation.{ext}',dpi=180)
    plt.close(fig)

    counts=pd.read_csv(results/'split_counts.csv').set_index('category')
    per=pd.read_csv(results/'per_class.csv').set_index(['condition','split','category'])
    rows=[]
    for name,_,_ in LABELS:
        c=counts.loc[name]
        rows.append(f"{name} & {int(c.pv_train):,} & {int(c.pv_val):,} & {int(c.pv_test):,} & {int(c.pd_test):,} \\\\")
    (dest/'counts.tex').write_text('\n'.join(rows)+'\n',encoding='utf-8')
    for metric in ['recall','f1']:
        rows=[]
        for name,_,_ in LABELS:
            values=[]
            for cond in ['ordinary','strong']:
                a=100*per.loc[(cond,'pv_test',name),metric]
                b=100*per.loc[(cond,'pd_test',name),metric]
                values.extend([a,b,a-b])
            rows.append(name+' & '+str(int(counts.loc[name,'pd_test']))+' & '+' & '.join(f'{v:.1f}' for v in values)+' \\\\')
        (dest/f'{metric}.tex').write_text('\n'.join(rows)+'\n',encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=Path('results'))
    p.add_argument('--dest',type=Path,default=Path('report/figures'))
    a=p.parse_args();figures(a.results,a.dest)
