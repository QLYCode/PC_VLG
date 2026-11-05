"""Fetch declared pretrained assets; verify MedSAM mirror digest; no SAM substitution."""
import hashlib,json,time,urllib.request,os
from pathlib import Path
if Path('/etc/ssl/certs/ca-certificates.crt').exists():
    os.environ.setdefault('SSL_CERT_FILE','/etc/ssl/certs/ca-certificates.crt')
from pathlib import Path


def fetch(url,target,expected=None):
    target=Path(target); target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() and (expected is None or hashlib.sha256(target.read_bytes()).hexdigest()==expected): return
    temporary=Path(str(target)+'.part')
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url,timeout=60) as src, temporary.open('wb') as dst:
                while True:
                    chunk=src.read(1024*1024)
                    if not chunk: break
                    dst.write(chunk)
            if expected and hashlib.sha256(temporary.read_bytes()).hexdigest()!=expected: raise ValueError('Checkpoint SHA256 mismatch')
            temporary.replace(target); print('Downloaded',target,flush=True); return
        except Exception as e:
            print('Download attempt',attempt+1,url,repr(e),flush=True)
            if attempt==2: raise
            time.sleep(5)


def main():
    root=Path(__file__).resolve().parent; assets=root/'assets'; assets.mkdir(exist_ok=True)
    sources={}
    try:
        url='https://huggingface.co/maregu2023/medsam-2d/resolve/main/medsam_vit_b.pth'
        sha='34b34b78c1d18cb8c6bf84cf9c00e135d6d6c965699f3c0e31ef1bc9dcb5be74'
        fetch(url,assets/'medsam_vit_b.pth',sha)
        sources['medsam']=dict(source=url,sha256=sha,origin='MedSAM converted checkpoint, public mirror')
    except Exception as e: sources['medsam_error']=repr(e)
    try:
        model='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract'
        with urllib.request.urlopen('https://huggingface.co/api/models/'+model,timeout=30) as r: meta=json.load(r)
        revision=meta['sha']; names={x['rfilename'] for x in meta['siblings']}
        files=['config.json','vocab.txt','pytorch_model.bin','tokenizer_config.json','special_tokens_map.json','tokenizer.json']
        for name in files:
            if name in names: fetch('https://huggingface.co/'+model+'/resolve/'+revision+'/'+name,assets/'pubmed_bert'/name)
        for name in ['config.json','vocab.txt','pytorch_model.bin']:
            if not (assets/'pubmed_bert'/name).is_file(): raise FileNotFoundError(name)
        sources['pubmed_bert']=dict(model=model,revision=revision)
        (assets/'pubmed_bert'/'DOWNLOAD_COMPLETE.json').write_text(json.dumps(sources['pubmed_bert'],indent=2))
    except Exception as e: sources['pubmed_bert_error']=repr(e)
    (assets/'asset_sources.json').write_text(json.dumps(sources,indent=2))
    if any(k.endswith('_error') for k in sources): raise SystemExit(1)

if __name__=='__main__': main()
