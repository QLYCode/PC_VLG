"""Prepare both patient-disjoint paper-sized splits; wait for verified FLARE archives."""
import argparse,json,random,subprocess,sys,time,zipfile,re,hashlib
from pathlib import Path


def run_prepare(dataset,split):
    root=Path(__file__).resolve().parent; out=root/'data'/dataset; out.mkdir(parents=True,exist_ok=True)
    sp=out/'patient_split.json'; sp.write_text(json.dumps(split,indent=2))
    subprocess.run([sys.executable,'prepare_pc_vlg_data.py','--split-json',str(sp),'--dataset',dataset,'--out',str(out)],cwd=str(root),check=True)


def acdc(raw_root,test_list_path):
    raw=Path(raw_root); images=sorted((raw/'images').glob('*.nii.gz'))
    patient={}
    for ip in images:
        name=ip.name[:-7]
        if not re.fullmatch(r'patient[0-9]{3}_frame[0-9]{2}',name): continue
        cid=name.split('_')[0]; label=raw/'full_annos'/ip.name
        if not label.exists(): raise FileNotFoundError(label)
        patient.setdefault(cid,[]).append(dict(case_id=cid,volume_id=name,image=str(ip),label=str(label)))
    if any(len(v)!=2 for v in patient.values()): raise ValueError('ACDC requires exactly ED/ES per patient')
    if len(patient)!=100: raise ValueError('Expected 100 ACDC patients, found '+str(len(patient)))
    # Preserve the existing 20-patient test split; repartition remaining 80 as 75/5.
    test_list=Path(test_list_path)
    test={x.split('_')[0] for x in test_list.read_text().split()}
    if len(test)!=20 or not test<=set(patient): raise ValueError('ACDC existing test list invalid')
    rest=sorted(set(patient)-test); random.Random(1337).shuffle(rest)
    membership=dict(train=rest[:75],val=rest[75:],test=sorted(test))
    split={s:[r for cid in membership[s] for r in patient[cid]] for s in membership}
    run_prepare('acdc',split)


def safe_extract(path,target):
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            dest=(target/info.filename).resolve()
            if target.resolve() not in dest.parents and dest!=target.resolve(): raise ValueError('Unsafe archive path')
        z.extractall(target)


def flare(raw_root):
    raw=Path(raw_root)
    while not (raw/'DOWNLOAD_VERIFIED').exists() and not (raw/'TRAINING_VERIFIED').exists():
        checks={'TrainingImg.zip':'3b231111a0cb5501396116f81a989511','TrainingMask.zip':'7373974f6e628b1631569bd2010cd233'}
        if all((raw/n).is_file() and zipfile.is_zipfile(raw/n) for n in checks):
            for name,want in checks.items():
                digest=hashlib.md5()
                with (raw/name).open('rb') as f:
                    for chunk in iter(lambda:f.read(8*1024*1024),b''): digest.update(chunk)
                if digest.hexdigest()!=want: raise ValueError('FLARE checksum mismatch '+name)
            (raw/'TRAINING_VERIFIED').touch()
            break
        print('FLARE21 waiting for verified training images/masks (unlabeled validation download is independent)',flush=True)
        time.sleep(60)
    for name in ['TrainingImg.zip','TrainingMask.zip']:
        marker=raw/(name+'.EXTRACTED')
        if not marker.exists(): safe_extract(raw/name,raw); marker.touch()
    files=list(raw.rglob('*.nii.gz'))
    images=[p for p in files if 'trainingimg' in str(p).lower()]
    masks=[p for p in files if 'trainingmask' in str(p).lower()]
    def key(p):
        x=p.name[:-7]
        if x.endswith('_0000'): x=x[:-5]
        return x
    maskmap={key(p):p for p in masks}
    pairs=[]
    for ip in images:
        cid=key(ip)
        if cid not in maskmap: raise ValueError('Missing FLARE21 label for '+cid)
        pairs.append(dict(case_id=cid,image=str(ip),label=str(maskmap[cid])))
    if len(pairs)!=361 or len({x['case_id'] for x in pairs})!=361: raise ValueError('Expected 361 labeled FLARE21 cases, found '+str(len(pairs)))
    pairs.sort(key=lambda r:r['case_id']); random.Random(1337).shuffle(pairs)
    run_prepare('flare21',dict(train=pairs[:260],val=pairs[260:286],test=pairs[286:]))


def main():
    p=argparse.ArgumentParser(); p.add_argument('dataset',choices=['acdc','flare21']); p.add_argument('--raw-root',required=True); p.add_argument('--acdc-test-list'); a=p.parse_args()
    if a.dataset=='acdc':
        if not a.acdc_test_list: p.error('--acdc-test-list is required for ACDC')
        acdc(a.raw_root,a.acdc_test_list)
    else: flare(a.raw_root)

if __name__=='__main__': main()
