"""Prepare explicit patient splits from axial NIfTI volumes; never invent split membership."""
import argparse,json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--split-json',required=True,help='train/val/test lists of {case_id,image,label}')
    p.add_argument('--dataset',choices=['flare21','acdc'],required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--allow-other-split',action='store_true')
    a=p.parse_args()
    import SimpleITK as sitk
    spec=json.loads(Path(a.split_json).read_text()); expected=(260,26,75) if a.dataset=='flare21' else (75,5,20)
    counts=tuple(len({str(r['case_id']) for r in spec[s]}) for s in ['train','val','test'])
    if not a.allow_other_split and counts!=expected: raise ValueError('Paper case counts {} != {}'.format(counts,expected))
    seen={}; volume_ids=set(); out=Path(a.out).resolve(); out.mkdir(parents=True,exist_ok=True)
    manifest={s:[] for s in ['train','val','test']}
    manifest['metadata']=dict(dataset=a.dataset,source_split=str(Path(a.split_json).resolve()),case_counts=counts,normalization='per volume minmax to [0,1]',size=256)
    for split in ['train','val','test']:
        for row in spec[split]:
            cid=str(row['case_id'])
            if cid in seen and seen[cid]!=split: raise ValueError('Patient leakage: '+cid)
            if '/' in cid or '\\' in cid or cid in ['.','..']: raise ValueError('Unsafe case identifier')
            seen[cid]=split
            vid=str(row.get('volume_id',cid))
            if vid in volume_ids or '/' in vid or '\\\\' in vid: raise ValueError('Duplicate/unsafe volume identifier')
            volume_ids.add(vid)
            im=sitk.ReadImage(row['image']); lab=sitk.ReadImage(row['label'])
            if im.GetSize()!=lab.GetSize() or not np.allclose(im.GetSpacing(),lab.GetSpacing()) or not np.allclose(im.GetDirection(),lab.GetDirection()) or not np.allclose(im.GetOrigin(),lab.GetOrigin()):
                raise ValueError('Image/label physical geometry mismatch: '+cid)
            x=sitk.GetArrayFromImage(im).astype(np.float32); y=sitk.GetArrayFromImage(lab).astype(np.int64)
            if not np.isfinite(x).all(): raise ValueError('Non-finite image: '+cid)
            nc=5 if a.dataset=='flare21' else 4
            if y.min()<0 or y.max()>=nc: raise ValueError('Explicit label mapping needed: '+cid)
            x=(x-x.min())/max(float(x.max()-x.min()),1e-8)
            from scipy.ndimage import zoom
            x=zoom(x,(1,256/x.shape[1],256/x.shape[2]),order=1,prefilter=False).astype(np.float32)
            y=y.astype(np.uint8)
            ip=out/(vid+'_image.npy'); lp=out/(vid+'_label.npy')
            np.save(ip,x); np.save(lp,y)
            for z in range(len(x)):
                manifest[split].append(dict(case_id=cid,volume_id=vid,image=str(ip),label=str(lp),slice=z,spacing=list(im.GetSpacing())[::-1]))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(out/'manifest.json')

if __name__=='__main__': main()
