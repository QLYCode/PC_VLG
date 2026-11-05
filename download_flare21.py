"""Official FLARE21 downloads with resumable wget and streaming MD5 verification."""
import argparse,hashlib,json,subprocess
from pathlib import Path

FILES={'TrainingImg.zip':'3b231111a0cb5501396116f81a989511','TrainingMask.zip':'7373974f6e628b1631569bd2010cd233','ValidationImg.zip':'a72cd1756bab0216731e53752d7af610'}
BASE='https://zenodo.org/records/5903672/files/'


def verify(path,want):
    digest=hashlib.md5()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): digest.update(chunk)
    if digest.hexdigest()!=want: raise ValueError('MD5 mismatch: '+str(path))


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',default='data/raw/FLARE21'); p.add_argument('--include-unlabeled-val',action='store_true'); p.add_argument('--verify-only',action='store_true')
    a=p.parse_args(); root=Path(a.root).resolve(); root.mkdir(parents=True,exist_ok=True)
    names=['TrainingMask.zip','TrainingImg.zip']+(['ValidationImg.zip'] if a.include_unlabeled_val else [])
    for name in names:
        target=root/name
        if not a.verify_only:
            subprocess.run(['wget','-c','--timeout=30','--tries=0','--waitretry=10','--retry-connrefused','-O',str(target),BASE+name+'?download=1'],check=True)
        verify(target,FILES[name]); print('MD5 OK',target,flush=True)
        if name=='TrainingImg.zip': (root/'TRAINING_VERIFIED').touch()
    if a.include_unlabeled_val: (root/'DOWNLOAD_VERIFIED').touch()
    (root/'download_sources.json').write_text(json.dumps(dict(record='https://zenodo.org/records/5903672',checksums={n:FILES[n] for n in names}),indent=2))

if __name__=='__main__': main()
