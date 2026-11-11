"""Matched-budget transfer learning with cached activations of the frozen prefix."""
import argparse
import csv
from pathlib import Path
import time
import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from sklearn.metrics import f1_score
from prepare import LABELS, write_csv
from paths import disk_path

SEED = 664
SIZE = 128
BATCH = 96
EPOCHS = 8
VIEWS = 2

def transforms(condition, training):
    ops = [T.Resize((SIZE,SIZE))]
    if training:
        if condition == 'strong':
            ops += [T.RandomResizedCrop(SIZE,scale=(.55,1.0),ratio=(.8,1.25)),
                    T.ColorJitter(brightness=.35,contrast=.35,saturation=.35,hue=.07)]
        ops.append(T.RandomHorizontalFlip())
    return T.Compose(ops+[T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])

class Leaves(Dataset):
    def __init__(self, rows, root, condition, training, view):
        self.rows, self.root, self.view = rows, root, view
        self.transform = transforms(condition,training)
    def __len__(self):
        return len(self.rows)
    def __getitem__(self,i):
        with Image.open(disk_path(self.root,self.rows[i]['path'])) as im:
            # Each training view is fixed across replays, workers and machines.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(SEED+100000*self.view+i)
                x=self.transform(ImageOps.exif_transpose(im).convert('RGB'))
        return x, int(self.rows[i]['label'])

def model(pretrained):
    net=mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
    net.classifier[-1]=nn.Linear(net.classifier[-1].in_features,len(LABELS))
    return net

@torch.inference_mode()
def encode(prefix,rows,root,condition,training,view,device):
    loader=DataLoader(Leaves(rows,root,condition,training,view),batch_size=BATCH,num_workers=0)
    feats=[]
    for x,_ in loader:
        feats.append(prefix(x.to(device=device,memory_format=torch.channels_last)).cpu().half())
    return torch.cat(feats)

def head(net, x):
    x=net.features[-2:](x)
    return net.classifier(net.avgpool(x).flatten(1))

@torch.inference_mode()
def predict(net,feats,device):
    net.eval()
    return torch.cat([head(net,x.to(device=device,dtype=torch.float32,memory_format=torch.channels_last)).softmax(1).cpu()
                      for x in feats.split(BATCH)])

def run(root,out,device):
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    rows=list(csv.DictReader(open(out/'manifest.csv')))
    sets={s:[r for r in rows if r['split']==s] for s in ('pv_train','pv_val','pv_test','pd_test')}
    y={s:torch.tensor([int(r['label']) for r in subset]) for s,subset in sets.items()}
    base=model(True).to(device=device,memory_format=torch.channels_last)
    initial={k:v.detach().cpu().clone() for k,v in base.state_dict().items()}
    prefix=base.features[:-2].eval()
    val=encode(prefix,sets['pv_val'],root,'ordinary',False,0,device)
    history=[]
    for condition in ('ordinary','strong'):
        tic=time.monotonic()
        train=[]
        for view in range(VIEWS):
            train.append(encode(prefix,sets['pv_train'],root,condition,True,view,device))
            print(condition,'encoded view',view,'seconds',round(time.monotonic()-tic),flush=True)
        net=model(False).to(device=device,memory_format=torch.channels_last)
        net.load_state_dict(initial)
        net.features[:-2].requires_grad_(False)
        opt=torch.optim.AdamW(filter(lambda p:p.requires_grad,net.parameters()),lr=.0005,weight_decay=.0001)
        rng=torch.Generator().manual_seed(SEED)
        torch.manual_seed(SEED)
        best=-1
        for epoch in range(EPOCHS):
            net.train()
            loss_sum=0.
            for ix in torch.randperm(len(y['pv_train']),generator=rng).split(BATCH):
                logits=head(net,train[epoch%VIEWS][ix].to(device=device,dtype=torch.float32,memory_format=torch.channels_last))
                loss=nn.functional.cross_entropy(logits,y['pv_train'][ix].to(device))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                loss_sum+=loss.item()*len(ix)
            yp=predict(net,val,device).argmax(1)
            target=y['pv_val']
            score=f1_score(target,yp,labels=target.unique().numpy(),average='macro',zero_division=0)
            history.append(dict(condition=condition,epoch=epoch+1,train_loss=loss_sum/len(y['pv_train']),
                                val_accuracy=(yp==target).float().mean().item(),val_supported_macro_f1=score))
            if score>best:
                best=score
                torch.save({'model':net.state_dict(),'condition':condition,'epoch':epoch+1,'size':SIZE,
                            'labels':[name for name,_,_ in LABELS]},out/f'{condition}.pt')
            print(condition,epoch+1,round(score,4),round(time.monotonic()-tic),flush=True)
        del train
    write_csv(out/'history.csv',history)
    # Both checkpoint choices are final before either test split is decoded.
    predictions=[]
    for s in ('pv_test','pd_test'):
        feats=encode(prefix,sets[s],root,'ordinary',False,0,device)
        for condition in ('ordinary','strong'):
            ckpt=torch.load(out/f'{condition}.pt',map_location=device,weights_only=True)
            net.load_state_dict(ckpt['model'])
            for r,probs in zip(sets[s],predict(net,feats,device).numpy()):
                predictions.append(dict(path=r['path'],split=s,condition=condition,y_true=r['label'],
                                        y_pred=int(np.argmax(probs)),confidence=float(max(probs)),
                                        **{f'p_{k}':float(p) for k,p in enumerate(probs)}))
    write_csv(out/'predictions.csv',predictions)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,default=Path('data'))
    p.add_argument('--out',type=Path,default=Path('results'))
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    args=p.parse_args()
    run(args.data,args.out,args.device)
