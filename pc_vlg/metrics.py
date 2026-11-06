"""Case-level coarse-network inference and physical-space segmentation metrics."""
import numpy as np
import torch
from medpy.metric.binary import dc,jc,hd95,asd
from .data import SliceDataset,load_npy_slice


@torch.no_grad()
def evaluate(coarse,rows,classes,device,surface_metrics=True):
    was_training=coarse.training; coarse.eval()
    ds=SliceDataset(rows)
    cases={}; counts={}
    for i,row in enumerate(rows):
        sample=ds[i]
        logits=coarse(sample['image'][None].to(device))
        if isinstance(logits,(tuple,list)): logits=logits[0]
        if not torch.isfinite(logits).all(): raise FloatingPointError('Non-finite coarse prediction')
        # Restore source geometry on-device, then accumulate overlap counts by case.
        pred=logits.argmax(1,keepdim=True).float()
        original=load_npy_slice(row['label'],int(row['slice']),np.int64)
        pred=torch.nn.functional.interpolate(pred,size=original.shape,mode='nearest')[0,0].long()
        gt=torch.from_numpy(original).to(device=device,dtype=torch.long)
        cid=str(row.get('volume_id',row['case_id']))
        bucket=cases.setdefault(cid,dict(spacing=row.get('spacing',[1,1,1]),slices=[]))
        if surface_metrics:
            bucket['slices'].append((int(row['slice']),pred.cpu().numpy(),original))
        pcount=torch.bincount(pred.reshape(-1),minlength=classes)
        gcount=torch.bincount(gt.reshape(-1),minlength=classes)
        inter=torch.bincount(pred[gt==pred].reshape(-1),minlength=classes)
        if cid not in counts:
            counts[cid]=[torch.zeros(classes,device=device,dtype=torch.float64) for _ in range(3)]
        counts[cid][0]+=inter; counts[cid][1]+=pcount; counts[cid][2]+=gcount
    records=[]
    for cid,(inter,pcount,gcount) in counts.items():
        for c in range(1,classes):
            denom=pcount[c]+gcount[c]; union=denom-inter[c]
            d=float((2*inter[c]/denom).item()) if denom else 1.
            j=float((inter[c]/union).item()) if union else 1.
            records.append(dict(case_id=cid,class_id=c,dsc=d,jac=j))
    if surface_metrics:
        # Surface distances remain CPU-only MedPy metrics and are optional because
        # their 3-D distance transforms dominate validation time on large volumes.
        for cid,item in cases.items():
            item['slices'].sort(key=lambda x:x[0])
            zs=[x[0] for x in item['slices']]
            if zs!=list(range(len(zs))): raise ValueError('Missing/duplicate slices in volume '+cid)
            pred=np.stack([x[1] for x in item['slices']]); gt=np.stack([x[2] for x in item['slices']])
            for c in range(1,classes):
                p=pred==c; g=gt==c
                if not p.any() and not g.any(): h,s=0.,0.
                elif not p.any() or not g.any(): h,s=None,None
                else: h,s=float(hd95(p,g,voxelspacing=item['spacing'])),float(asd(p,g,voxelspacing=item['spacing']))
                rec=next(x for x in records if x['case_id']==cid and x['class_id']==c)
                rec.update(hd95_mm=h,asd_mm=s)
    summary={}
    for key in ['dsc','jac']:
        v=[r[key] for r in records if r[key] is not None]
        summary[key]=float(np.mean(v)) if v else None
        summary[key+'_valid']=len(v)
    summary['per_class']={str(c):{k:float(np.mean([r[k] for r in records if r['class_id']==c and r[k] is not None])) if any(r['class_id']==c and r[k] is not None for r in records) else None for k in ['dsc','jac']} for c in range(1,classes)}
    if surface_metrics:
        for key in ['hd95_mm','asd_mm']:
            v=[r[key] for r in records if r.get(key) is not None]
            summary[key]=float(np.mean(v)) if v else None
            summary[key+'_valid']=len(v)
        for c in range(1,classes):
            for key in ['hd95_mm','asd_mm']:
                v=[r[key] for r in records if r['class_id']==c and r.get(key) is not None]
                summary['per_class'][str(c)][key]=float(np.mean(v)) if v else None
    summary['cases']=len(cases); summary['per_case']=records
    coarse.train(was_training)
    return summary
