"""Patient-separated slice loading. Unlabeled masks are never read by __getitem__."""
import json
import math
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset,Sampler
from torch.nn import functional as F


def audit_manifest(manifest,ratio=0.1,seed=1337):
    m=json.loads(Path(manifest).read_text())
    pools={split:{str(r['case_id']) for r in m[split]} for split in ['train','val','test']}
    for a,b in [('train','val'),('train','test'),('val','test')]:
        if pools[a]&pools[b]: raise ValueError('case leakage between '+a+' and '+b)
    cases=sorted(pools['train']); random.Random(seed).shuffle(cases)
    n=max(1,int(len(cases)*ratio))
    if n>=len(cases): raise ValueError('SSL requires both labeled and unlabeled patients')
    labeled=set(cases[:n])
    return m,labeled


class SliceDataset(Dataset):
    def __init__(self,rows,labeled_cases=None,augment=False,size=256):
        self.rows=rows; self.labeled_cases=labeled_cases; self.augment=augment; self.size=size
    def __len__(self): return len(self.rows)
    def __getitem__(self,index):
        r=self.rows[index]; z=int(r['slice'])
        image=np.asarray(np.load(r['image'],mmap_mode='r')[z],dtype=np.float32).copy()
        labeled=self.labeled_cases is None or str(r['case_id']) in self.labeled_cases
        label=np.asarray(np.load(r['label'],mmap_mode='r')[z],dtype=np.int64).copy() if labeled else np.full(image.shape,-1,dtype=np.int64)
        x=torch.from_numpy(image)[None,None]
        x=F.interpolate(x,size=(self.size,self.size),mode='bilinear',align_corners=False).clamp(0,1)
        y=F.interpolate(torch.from_numpy(label).float()[None,None],size=(self.size,self.size),mode='nearest')
        if 'midline' in r: midline=float(r['midline'])
        else:
            body=torch.where(x[0,0]>0.05)[1]
            midline=float((body.min()+body.max())/(2*(self.size-1))) if len(body) else 0.5
        theta=torch.tensor([[1.,0,0],[0,1.,0]])
        if self.augment:
            angle=random.uniform(-20,20)*math.pi/180; zoom=random.uniform(0.8,1.2)
            fx=-1. if random.random()<0.5 else 1.; fy=-1. if random.random()<0.5 else 1.
            theta=torch.tensor([[fx*math.cos(angle)/zoom,-fy*math.sin(angle)/zoom,0],
                                [fx*math.sin(angle)/zoom,fy*math.cos(angle)/zoom,0]])
            grid=F.affine_grid(theta[None],x.shape,align_corners=True)
            x=F.grid_sample(x,grid,mode='bilinear',align_corners=True)
            y=F.grid_sample(y,grid,mode='nearest',align_corners=True,padding_mode='border')
        return dict(image=x[0],label=y[0,0].long(),labeled=labeled,midline=midline,affine=theta,
                    case_id=str(r['case_id']),slice=z,spacing=r.get('spacing',[1,1,1]))


class TwoStreamBatchSampler(Sampler):
    def __init__(self,rows,labeled_cases,batch_size=4,labeled_bs=2,seed=1337):
        self.labeled=[i for i,r in enumerate(rows) if str(r['case_id']) in labeled_cases]
        self.unlabeled=[i for i,r in enumerate(rows) if str(r['case_id']) not in labeled_cases]
        self.bs=batch_size; self.lb=labeled_bs; self.seed=seed; self.epoch=0
        if not 0<labeled_bs<batch_size or not self.labeled or not self.unlabeled:
            raise ValueError('batch needs nonempty labeled and unlabeled streams')
    def __len__(self): return max(1,math.ceil(len(self.unlabeled)/(self.bs-self.lb)))
    def __iter__(self):
        rng=random.Random(self.seed+self.epoch); self.epoch+=1
        def stream(pool):
            while True:
                order=pool.copy(); rng.shuffle(order)
                yield from order
        l,u=stream(self.labeled),stream(self.unlabeled)
        for _ in range(len(self)):
            yield [next(l) for _ in range(self.lb)]+[next(u) for _ in range(self.bs-self.lb)]
