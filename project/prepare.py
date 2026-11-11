"""Build the nine-class file manifest and audit identical source bytes."""
from collections import defaultdict
from pathlib import Path
import argparse
import csv
import hashlib
import json
import random
import re
from paths import disk_path, canonical_path, storage_root

# Order is the class index stored in each checkpoint.
LABELS = [
    ("Healthy", "healthy", "Tomato leaf"),
    ("Early blight", "Early_blight", "Tomato Early blight leaf"),
    ("Bacterial spot", "Bacterial_spot", "Tomato leaf bacterial spot"),
    ("Late blight", "Late_blight", "Tomato leaf late blight"),
    ("Mosaic virus", "Tomato_mosaic_virus", "Tomato leaf mosaic virus"),
    ("Yellow leaf curl", "Tomato_Yellow_Leaf_Curl_Virus", "Tomato leaf yellow virus"),
    ("Leaf mold", "Leaf_Mold", "Tomato mold leaf"),
    ("Septoria", "Septoria_leaf_spot", "Tomato Septoria leaf spot"),
    ("Spider mites", "Spider_mites Two-spotted_spider_mite", "Tomato two spotted spider mites leaf"),
]
def norm(s):
    return re.sub('[^a-z0-9]+', '', s.casefold())

def write_csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def prepare(root, out):
    root=storage_root(root)
    pv = root / 'PlantVillage-Dataset-master'
    pd = root / 'PlantDoc-Dataset-master'
    leafmap = json.loads((root / 'leaf-map.json').read_text(encoding='utf-8'))
    source = {norm('Tomato___'+v): i for i, (_, v, _) in enumerate(LABELS)}
    field = {norm(v): i for i, (_, _, v) in enumerate(LABELS)}
    rows, excluded = [], []
    def add(path, domain, upstream, split, k, group=''):
        rows.append(dict(path=canonical_path(path,root), domain=domain,
                         upstream_split=upstream, split=split, label=k, category=LABELS[k][0],
                         leaf_group=group, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    for split in ('train', 'test'):
        for line in (root / f'color_{split}.txt').read_text(encoding='utf-8').splitlines():
            path = disk_path(root, 'PlantVillage-Dataset-master/'+line.strip())
            folder = path.parent.name
            k = source.get(norm(folder))
            if k is None:
                continue
            key = path.stem.split('___')[-1].strip().lower()
            ids = [g for g in leafmap.get(key, []) if g.startswith(folder+':::')]
            if len(ids) > 1:
                raise ValueError(f'Ambiguous leaf id: {path}')
            add(path, 'PlantVillage', split, 'pv_'+split, k, ids[0] if ids else '')
    # Only documented leaf IDs can enter validation. Unknown IDs stay in training.
    rng = random.Random(664)
    for k in range(len(LABELS)):
        groups = sorted({r['leaf_group'] for r in rows if r['label']==k
                         and r['split']=='pv_train' and r['leaf_group']})
        rng.shuffle(groups)
        holdout = set(groups[:round(0.15*len(groups))])
        for r in rows:
            if r['split']=='pv_train' and r['leaf_group'] in holdout:
                r['split']='pv_val'
    for split in ('train','test'):
        for path in sorted((pd / split).glob('*/*'),key=lambda p:canonical_path(p,root)):
            k = field.get(norm(path.parent.name))
            if k is not None and path.is_file() and path.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.webp','.gif','.jfif','.tif','.tiff'}:
                add(path,'PlantDoc',split,'pd_test',k)
    by_hash = defaultdict(list)
    for r in rows:
        by_hash[r['sha256']].append(r)
    remove = set()
    for batch in by_hash.values():
        if len({r['split'] for r in batch})>1:
            priority = {'pv_train':0, 'pv_val':1, 'pv_test':2, 'pd_test':3}
            high = max(priority[r['split']] for r in batch)
            for r in batch:
                if priority[r['split']] < high:
                    excluded.append(dict(**r, reason='identical bytes in a held-out split'))
                    remove.add(r['path'])
        if batch[0]['domain']=='PlantDoc' and len(batch)>1:
            conflict=len({r['label'] for r in batch})>1
            for r in (batch if conflict else sorted(batch,key=lambda x:x['path'])[1:]):
                excluded.append(dict(**r, reason='conflicting labels' if conflict else 'exact duplicate'))
                remove.add(r['path'])
    rows=sorted((r for r in rows if r['path'] not in remove),key=lambda r:r['path'])
    seen={}
    for r in rows:
        g=r['leaf_group']
        if g and seen.setdefault(g,r['split']) != r['split']:
            raise ValueError(f'Leaf {g} crosses a split')
    for split in ('pv_train','pv_test','pd_test'):
        assert len({r['label'] for r in rows if r['split']==split})==9
    assert len({r['path'] for r in rows})==len(rows)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out/'manifest.csv',rows)
    if excluded:
        write_csv(out/'exclusions.csv',excluded)
    write_csv(out/'class_mapping.csv',[dict(label=i,category=name,PlantVillage='Tomato___'+pvname,PlantDoc=pdname)
                                        for i,(name,pvname,pdname) in enumerate(LABELS)])
    from collections import Counter
    print(Counter(r['split'] for r in rows), 'exclusions',len(excluded))
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=Path('data'))
    parser.add_argument('--out',type=Path,default=Path('results'))
    args=parser.parse_args()
    prepare(args.data,args.out)
