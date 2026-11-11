"""Compare top predictions on one photo with both shipped checkpoints, offline."""
import argparse
from pathlib import Path
from PIL import Image, ImageOps
import torch
from experiment import model, transforms, head

def classify(image, results):
    torch.set_num_threads(2)
    with Image.open(image) as im:
        x=transforms('ordinary',False)(ImageOps.exif_transpose(im).convert('RGB')).unsqueeze(0)
    for name in ['ordinary','strong']:
        ckpt=torch.load(results/f'{name}.pt',map_location='cpu',weights_only=True)
        net=model(False)
        net.load_state_dict(ckpt['model'])
        net.eval()
        with torch.inference_mode():
            z=net.features[:-2](x).half().float()
            probs=head(net,z).softmax(1)[0]
        print(f'{name} (seed {ckpt.get("seed", "legacy")}, epoch {ckpt["epoch"]})')
        for i in probs.argsort(descending=True)[:3]:
            print(f'  {ckpt["labels"][i]:22} {float(probs[i]):.1%}')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('photo',type=Path)
    p.add_argument('--results',type=Path,default=Path(__file__).resolve().parent/'results/fresh_views/seed_664')
    args=p.parse_args()
    classify(args.photo,args.results)
