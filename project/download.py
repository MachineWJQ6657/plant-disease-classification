"""Retrieve pinned upstream snapshots. Keep just the required tomato color folders."""
import argparse
from pathlib import Path, PurePosixPath
import shutil
from urllib.request import urlopen
from zipfile import ZipFile
from prepare import LABELS, norm
from paths import disk_path, storage_root

PV='7f7ecc7e1eaca78107e3affe7cb5abd9427e139a'
PD='5467f6012d78d1c446145d5f582da6096f852ae8'
HF='9e97599868962bd0079b8db4b7f1efa9185fa1e7'

def download(url,path):
    if path.exists():
        return
    temp=path.with_suffix('.part')
    with urlopen(url,timeout=180) as response,temp.open('wb') as f:
        shutil.copyfileobj(response,f)
    temp.replace(path)

def get_data(root):
    root=storage_root(root)
    root.mkdir(parents=True,exist_ok=True)
    pv_classes={norm('Tomato___'+pv) for _,pv,_ in LABELS}
    pd_classes={norm(pd) for _,_,pd in LABELS}
    for name,commit,repo,prefix in [
        ('pv',PV,'spMohanty/PlantVillage-Dataset','PlantVillage-Dataset-master'),
        ('pd',PD,'pratikkayal/PlantDoc-Dataset','PlantDoc-Dataset-master')]:
        archive=root/(name+'.zip')
        download(f'https://codeload.github.com/{repo}/zip/{commit}',archive)
        with ZipFile(archive) as z:
            if z.comment.decode().strip()!=commit:
                raise ValueError(f'Unexpected snapshot in {archive}. Remove it and retry.')
            for info in z.infolist():
                p=PurePosixPath(info.filename)
                parts=p.parts[1:]
                if info.is_dir() or '..' in parts:
                    continue
                pv_match=len(parts)==4 and parts[:2]==('raw','color') and norm(parts[2]) in pv_classes
                pd_match=len(parts)==3 and parts[0] in ('train','test') and norm(parts[1]) in pd_classes
                if (name=='pv' and pv_match) or (name=='pd' and pd_match):
                    target=disk_path(root,PurePosixPath(prefix,*parts).as_posix())
                    target.parent.mkdir(parents=True,exist_ok=True)
                    with z.open(info) as src,target.open('wb') as dest:
                        shutil.copyfileobj(src,dest)
    download(f'https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/{PV}/leaf_grouping/leaf-map.json',
             root/'leaf-map.json')
    for split in ('train','test'):
        filename=f'color_{split}.txt'
        download(f'https://huggingface.co/datasets/mohanty/PlantVillage/resolve/{HF}/splits/{filename}',root/filename)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,default=Path('data'))
    get_data(p.parse_args().data)
