"""Case-level coarse-network inference and physical-space segmentation metrics."""
import numpy as np
import torch
from medpy.metric.binary import dc,jc,hd95,asd
from .data import SliceDataset,load_npy_slice


@torch.no_grad()
def evaluate(coarse,rows,classes,device):
    was_training=coarse.training; coarse.eval()
    ds=SliceDataset(rows)
    cases={}
    for i,row in enumerate(rows):
        sample=ds[i]
        logits=coarse(sample['image'][None].to(device))
        if isinstance(logits,(tuple,list)): logits=logits[0]
        if not torch.isfinite(logits).all(): raise FloatingPointError('Non-finite coarse prediction')
        pred=logits.argmax(1)[0].cpu().numpy()
        gt=sample['label'].numpy()
        # Restore slice geometry before volume metrics (nearest neighbor).
        original=load_npy_slice(row['label'],int(row['slice']),np.int64)
        pred=torch.nn.functional.interpolate(torch.from_numpy(pred)[None,None].float(),size=original.shape,mode='nearest')[0,0].long().numpy()
        bucket=cases.setdefault(str(row.get('volume_id',row['case_id'])),dict(slices=[],spacing=row.get('spacing',[1,1,1])))
        bucket['slices'].append((int(row['slice']),pred,np.asarray(original)))
    records=[]
    for cid,item in cases.items():
        item['slices'].sort(key=lambda x:x[0])
        zs=[x[0] for x in item['slices']]
        if zs!=list(range(len(zs))): raise ValueError('Missing/duplicate slices in volume '+cid)
        pred=np.stack([x[1] for x in item['slices']]); gt=np.stack([x[2] for x in item['slices']])
        for c in range(1,classes):
            p=pred==c; g=gt==c
            if not p.any() and not g.any(): d,j,h,s=1.,1.,0.,0.
            elif not p.any() or not g.any(): d,j,h,s=0.,0.,None,None
            else: d,j,h,s=float(dc(p,g)),float(jc(p,g)),float(hd95(p,g,voxelspacing=item['spacing'])),float(asd(p,g,voxelspacing=item['spacing']))
            records.append(dict(case_id=cid,class_id=c,dsc=d,jac=j,hd95_mm=h,asd_mm=s))
    summary={}
    for key in ['dsc','jac','hd95_mm','asd_mm']:
        v=[r[key] for r in records if r[key] is not None]
        summary[key]=float(np.mean(v)) if v else None
        summary[key+'_valid']=len(v)
    summary['per_class']={str(c):{k:float(np.mean([r[k] for r in records if r['class_id']==c and r[k] is not None])) if any(r['class_id']==c and r[k] is not None for r in records) else None for k in ['dsc','jac','hd95_mm','asd_mm']} for c in range(1,classes)}
    summary['cases']=len(cases); summary['per_case']=records
    coarse.train(was_training)
    return summary
