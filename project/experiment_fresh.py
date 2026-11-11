"""Three paired seeds with fresh per-image PIL augmentation each epoch.

The original two-view experiment.py is retained unchanged. This extension uses
the same split, architecture, trainable blocks, image size, optimizer and budget.
All six checkpoints are selected on source validation before either test is scored.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import time

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode
import torchvision
from sklearn.metrics import f1_score

from experiment import model, head, predict, SIZE, BATCH, EPOCHS
from prepare import LABELS, write_csv
from paths import disk_path


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest_state(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.part')
    temp.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    temp.replace(path)


def event(out,kind,**values):
    row=dict(time_utc=utc(),event=kind,**values)
    with (out/'events.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)


def crop_parameters(rng,width,height):
    area=width*height
    for _ in range(10):
        target=rng.uniform(.55,1.)*area
        aspect=math.exp(rng.uniform(math.log(.8),math.log(1.25)))
        w=round(math.sqrt(target*aspect));h=round(math.sqrt(target/aspect))
        if 0<w<=width and 0<h<=height:
            return rng.randint(0,height-h),rng.randint(0,width-w),h,w
    # Same central fallback rule as torchvision RandomResizedCrop.
    ratio=width/height
    if ratio<.8:
        w=width;h=round(w/.8)
    elif ratio>1.25:
        h=height;w=round(h*1.25)
    else:
        w=width;h=height
    return (height-h)//2,(width-w)//2,h,w


def fresh_transform(rgb,condition,seed,epoch,index):
    """Local RNGs make views reproducible across threads, with new draws each epoch."""
    rng=random.Random(int(seed)*1_000_000_000+int(epoch)*100_000+int(index))
    im=Image.fromarray(np.asarray(rgb,dtype=np.uint8))
    if condition=='strong':
        top,left,height,width=crop_parameters(rng,*im.size)
        im=F.resized_crop(im,top,left,height,width,[SIZE,SIZE],InterpolationMode.BILINEAR,antialias=True)
        factors=[rng.uniform(.65,1.35),rng.uniform(.65,1.35),rng.uniform(.65,1.35),rng.uniform(-.07,.07)]
        order=list(range(4));rng.shuffle(order)
        functions=[F.adjust_brightness,F.adjust_contrast,F.adjust_saturation,F.adjust_hue]
        for k in order:
            im=functions[k](im,factors[k])
    elif condition!='ordinary':
        raise ValueError('Unknown condition')
    if rng.random()<.5:
        im=F.hflip(im)
    return F.normalize(F.to_tensor(im),[.485,.456,.406],[.229,.224,.225])


def prepare_pixels(rows,data,cache,workers,out,split_name):
    """Cache only deterministic RGB resize. Random training views are never cached."""
    descriptor=json.dumps(dict(paths=[(r['path'],r['sha256']) for r in rows],size=SIZE,
                               interpolation='PIL bilinear',exif=True),sort_keys=True).encode()
    key=hashlib.sha256(descriptor).hexdigest()
    cache.mkdir(parents=True,exist_ok=True)
    pixel_path=cache/(split_name+'-'+key[:16]+'.npy')
    receipt=pixel_path.with_suffix('.json')
    if receipt.exists() and pixel_path.exists():
        record=json.loads(receipt.read_text(encoding='utf-8'))
        if record['descriptor_sha256']!=key:
            raise ValueError('Pixel-cache descriptor mismatch')
        event(out,'pixel_cache_reused',split=split_name,images=len(rows))
        return np.load(pixel_path,mmap_mode='r')
    pixels=np.lib.format.open_memmap(pixel_path,mode='w+',dtype=np.uint8,shape=(len(rows),SIZE,SIZE,3))
    def decode(item):
        i,r=item
        raw=disk_path(data,r['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=r['sha256']:
            raise ValueError('Raw image hash mismatch: '+r['path'])
        with Image.open(io.BytesIO(raw)) as im:
            rgb=ImageOps.exif_transpose(im).convert('RGB')
            rgb=F.resize(rgb,[SIZE,SIZE],InterpolationMode.BILINEAR,antialias=True)
            return i,np.asarray(rgb)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0,len(rows),256):
            items=list(enumerate(rows[start:start+256],start=start))
            for i,rgb in pool.map(decode,items):
                pixels[i]=rgb
            if start%2048==0:
                event(out,'pixel_cache_progress',split=split_name,completed=min(start+256,len(rows)),total=len(rows))
    pixels.flush();del pixels
    write_json(receipt,dict(descriptor_sha256=key,images=len(rows),raw_sha256_verified=True,created_utc=utc()))
    event(out,'pixel_cache_complete',split=split_name,images=len(rows),all_raw_hashes_verified=True)
    return np.load(pixel_path,mmap_mode='r')


@torch.inference_mode()
def encode_pixels(prefix,pixels,device):
    blocks=[]
    mean=torch.tensor([.485,.456,.406],device=device).view(1,3,1,1)
    std=torch.tensor([.229,.224,.225],device=device).view(1,3,1,1)
    for start in range(0,len(pixels),BATCH):
        x=torch.from_numpy(np.array(pixels[start:start+BATCH],copy=True)).permute(0,3,1,2)
        x=x.to(device=device,dtype=torch.float32,memory_format=torch.channels_last)/255.
        x=(x-mean)/std
        blocks.append(prefix(x).cpu().half())
    return torch.cat(blocks)


def run_fresh(data,manifest_path,out,cache,device,seeds,workers=4,threads=4):
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    with manifest_path.open(encoding='utf-8',newline='') as f:
        rows=list(csv.DictReader(f))
    sets={split:[r for r in rows if r['split']==split] for split in ['pv_train','pv_val','pv_test','pd_test']}
    script_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    plan=dict(protocol='fresh_views_v1',seeds=seeds,conditions=['ordinary','strong'],epochs=EPOCHS,batch_size=BATCH,
        input_size=SIZE,initialization='ImageNet MobileNetV3-Small; identical within seed',
        trainable='last inverted residual block, final convolution, classifier',
        optimizer='AdamW',learning_rate=.0005,weight_decay=.0001,
        fresh_view='independent per-image local Python RNG each epoch; PIL transforms',
        random_seed_formula='seed*1000000000 + epoch*100000 + image_index',
        ordinary='PIL resize 128, p=0.5 horizontal flip, ImageNet normalization',
        strong='same plus crop area .55-1, log-uniform ratio .8-1.25 and jitter .35/.35/.35/.07',
        prefix='eval mode, frozen weights, fresh forward pass each batch; float16 activations then float32 trainable blocks',
        checkpoint_selection='first maximum eight-supported-class source-validation macro-F1',
        split_selection='original fixed split; no seed-based resplitting',
        external_evaluation='only after all six checkpoints finalized; no field-based selection',
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),script_sha256=script_hash,
        torch=torch.__version__,torchvision=torchvision.__version__,python=platform.python_version(),
        hardware=torch.cuda.get_device_name(0) if device.startswith('cuda') else platform.processor(),
        device=device,deterministic_algorithms=True,tf32=False)
    plan_file=out/'protocol.json'
    if plan_file.exists():
        if json.loads(plan_file.read_text(encoding='utf-8'))!=plan:
            raise ValueError('Output contains a different protocol. Use a new output directory.')
    else:
        write_json(plan_file,plan)
    event(out,'run_started',seeds=seeds,device=device,script_sha256=script_hash)
    train_pixels=prepare_pixels(sets['pv_train'],data,cache,workers,out,'pv_train')
    val_pixels=prepare_pixels(sets['pv_val'],data,cache,workers,out,'pv_val')
    y_train=torch.tensor([int(r['label']) for r in sets['pv_train']],device=device)
    y_val=np.array([int(r['label']) for r in sets['pv_val']])
    torch.manual_seed(seeds[0])
    base=model(True).to(device=device,memory_format=torch.channels_last)
    prefix=base.features[:-2].eval()
    prefix.requires_grad_(False)
    prefix_hash=digest_state(prefix.state_dict())
    val_features=encode_pixels(prefix,val_pixels,device)
    for seed in seeds:
        seed_out=out/f'seed_{seed}';seed_out.mkdir(exist_ok=True)
        shutil.copy2(manifest_path,seed_out/'manifest.csv')
        for name in ['class_mapping.csv','exclusions.csv']:
            source=manifest_path.parent/name
            if source.exists():shutil.copy2(source,seed_out/name)
        torch.manual_seed(seed)
        net=model(True).to(device=device,memory_format=torch.channels_last)
        initial={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
        initial_hash=digest_state(initial)
        del net
        for condition in ['ordinary','strong']:
            done=seed_out/f'{condition}_training_complete.json'
            if done.exists():
                event(out,'training_already_complete',seed=seed,condition=condition)
                continue
            net=model(False).to(device=device,memory_format=torch.channels_last)
            net.load_state_dict(initial)
            net.features[:-2].requires_grad_(False)
            opt=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=.0005,weight_decay=.0001)
            history_file=seed_out/f'history_{condition}.csv'
            records=[];best=-1.;start_epoch=0
            resume_path=cache/f'resume_{seed}_{condition}.pt'
            if resume_path.exists():
                resume=torch.load(resume_path,map_location=device,weights_only=True)
                if resume['script_sha256']!=script_hash:raise ValueError('Resume source mismatch')
                net.load_state_dict(resume['model']);opt.load_state_dict(resume['optimizer'])
                records=resume['records'];best=resume['best'];start_epoch=resume['epoch']
                torch.set_rng_state(resume['cpu_rng'].cpu())
                if device.startswith('cuda'):torch.cuda.set_rng_state(resume['cuda_rng'].cpu())
                del resume
            else:
                torch.manual_seed(seed)
                if device.startswith('cuda'):torch.cuda.manual_seed_all(seed)
            event(out,'condition_started',seed=seed,condition=condition,initial_state_sha256=initial_hash,start_epoch=start_epoch)
            condition_start=time.monotonic()
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for epoch in range(start_epoch,EPOCHS):
                    epoch_start=time.monotonic()
                    net.train();net.features[:-2].eval()
                    order=torch.randperm(len(y_train),generator=torch.Generator().manual_seed(seed+epoch*1_000_000))
                    order_hash=hashlib.sha256(order.numpy().tobytes()).hexdigest()
                    loss_sum=0.;step=0
                    for ix in order.split(BATCH):
                        indices=ix.tolist()
                        def augment(i):
                            return fresh_transform(train_pixels[i],condition,seed,epoch,i)
                        x=torch.stack(list(pool.map(augment,indices))).to(device=device,memory_format=torch.channels_last)
                        with torch.no_grad():
                            features=net.features[:-2](x).half().float()
                        logits=head(net,features)
                        loss=nn.functional.cross_entropy(logits,y_train[ix.to(device)])
                        if not torch.isfinite(loss):raise ValueError('Non-finite training loss')
                        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
                        loss_sum+=float(loss.detach())*len(ix);step+=1
                    probs=predict(net,val_features,device).numpy()
                    pred=probs.argmax(1)
                    score=float(f1_score(y_val,pred,labels=np.unique(y_val),average='macro',zero_division=0))
                    row=dict(condition=condition,epoch=epoch+1,train_loss=loss_sum/len(y_train),
                        val_accuracy=float((pred==y_val).mean()),val_supported_macro_f1=score,
                        seconds=round(time.monotonic()-epoch_start,3),optimizer_steps=step,
                        sample_order_sha256=order_hash,augmentation_epoch=epoch)
                    records.append(row)
                    if score>best:
                        best=score
                        checkpoint=dict(model={k:v.detach().cpu().clone() for k,v in net.state_dict().items()},
                            condition=condition,epoch=epoch+1,size=SIZE,labels=[x[0] for x in LABELS],seed=seed,
                            protocol='fresh_views_v1',initial_state_sha256=initial_hash,
                            source_prefix_sha256=prefix_hash,script_sha256=script_hash)
                        torch.save(checkpoint,seed_out/f'{condition}.pt')
                        del checkpoint
                    write_csv(history_file,records)
                    resume=dict(model=net.state_dict(),optimizer=opt.state_dict(),records=records,best=best,
                        epoch=epoch+1,cpu_rng=torch.get_rng_state(),script_sha256=script_hash)
                    if device.startswith('cuda'):resume['cuda_rng']=torch.cuda.get_rng_state()
                    temp=resume_path.with_suffix('.part');torch.save(resume,temp);temp.replace(resume_path)
                    event(out,'epoch_complete',seed=seed,**row)
            assert digest_state(net.features[:-2].state_dict())==prefix_hash
            final_hash=digest_state(net.state_dict())
            assert final_hash!=initial_hash
            write_json(done,dict(seed=seed,condition=condition,epochs=EPOCHS,optimizer_steps=sum(r['optimizer_steps'] for r in records),
                image_presentations=len(y_train)*EPOCHS,initial_state_sha256=initial_hash,final_state_sha256=final_hash,
                frozen_prefix_unchanged=True,training_seconds=round(time.monotonic()-condition_start,3),completed_utc=utc()))
            del net,opt
        combined=[]
        for condition in ['ordinary','strong']:
            with (seed_out/f'history_{condition}.csv').open(encoding='utf-8',newline='') as f:
                combined.extend(csv.DictReader(f))
        write_csv(seed_out/'history.csv',combined)
    # This milestone is written before any test images are read or test predictions produced.
    write_json(out/'selection_locked.json',dict(time_utc=utc(),seeds=seeds,all_conditions_complete=True,
        checkpoint_sha256={f'{seed}/{condition}':hashlib.sha256((out/f'seed_{seed}'/f'{condition}.pt').read_bytes()).hexdigest()
                           for seed in seeds for condition in ['ordinary','strong']}))
    event(out,'all_checkpoints_locked_before_test_evaluation')
    for split in ['pv_test','pd_test']:
        pixels=prepare_pixels(sets[split],data,cache,workers,out,split)
        feats=encode_pixels(prefix,pixels,device)
        for seed in seeds:
            seed_out=out/f'seed_{seed}'
            predictions=[]
            for condition in ['ordinary','strong']:
                checkpoint=torch.load(seed_out/f'{condition}.pt',map_location='cpu',weights_only=True)
                net=model(False).to(device=device,memory_format=torch.channels_last)
                net.load_state_dict(checkpoint['model']);net.eval()
                assert digest_state(net.features[:-2].state_dict())==prefix_hash
                for r,p in zip(sets[split],predict(net,feats,device).numpy()):
                    predictions.append(dict(path=r['path'],split=split,condition=condition,y_true=r['label'],y_pred=int(p.argmax()),
                        confidence=float(p.max()),**{f'p_{k}':float(v) for k,v in enumerate(p)}))
                del net,checkpoint
            write_csv(seed_out/f'predictions_{split}.csv',predictions)
            event(out,'test_predictions_saved',seed=seed,split=split,predictions=len(predictions))
        del pixels,feats
    for seed in seeds:
        seed_out=out/f'seed_{seed}'
        combined=[]
        for split in ['pv_test','pd_test']:
            with (seed_out/f'predictions_{split}.csv').open(encoding='utf-8',newline='') as f:
                combined.extend(csv.DictReader(f))
        write_csv(seed_out/'predictions.csv',combined)
        from analyze import analyze
        from verify import audit
        analyze(seed_out)
        write_json(seed_out/'artifact_verification.json',audit(seed_out))
    event(out,'run_complete',seeds=seeds)
    write_json(out/'run_complete.json',dict(status='PASS',completed_utc=utc(),trained_models=2*len(seeds),
        epochs_per_model=EPOCHS,total_optimizer_steps=2*len(seeds)*math.ceil(len(y_train)/BATCH)*EPOCHS,
        total_image_presentations=2*len(seeds)*len(y_train)*EPOCHS,all_artifact_audits_passed=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--manifest',type=Path,default=Path('results/manifest.csv'))
    p.add_argument('--out',type=Path,default=Path('results/fresh_views'))
    p.add_argument('--cache',type=Path,default=Path('.build/fresh-cache'))
    p.add_argument('--device',default='cuda')
    p.add_argument('--seeds',type=int,nargs='+',default=[664,665,666])
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--threads',type=int,default=4)
    a=p.parse_args()
    if len(a.seeds)!=len(set(a.seeds)):p.error('Seeds must be distinct')
    if a.out.resolve()==a.manifest.parent.resolve():p.error('Use a separate output directory')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    run_fresh(a.data,a.manifest,a.out,a.cache,a.device,a.seeds,a.workers,a.threads)
