import argparse,json,os,subprocess,sys,time,fcntl
from pathlib import Path


def main():
    p=argparse.ArgumentParser(); p.add_argument('--gpu',default='3'); p.add_argument('--poll',type=int,default=300); p.add_argument('--once',action='store_true'); p.add_argument('--stable-acdc',action='store_true')
    a=p.parse_args(); root=Path(__file__).resolve().parent; os.chdir(root)
    out=root/'runs'/'queue_gpu3'; out.mkdir(parents=True,exist_ok=True)
    lock=(out/'queue.lock').open('w')
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise SystemExit('GPU3 PC-VLG queue already running')
    jobs=([dict(dataset='acdc',ratio=r,stable=True) for r in [0.1,0.2]]+[dict(dataset='flare21',ratio=r) for r in [0.1,0.2]]) if a.stable_acdc else [dict(dataset=d,ratio=r) for d in ['flare21','acdc'] for r in [0.1,0.2]]
    def job_key(job): return '{}{}_label{}'.format(job['dataset'],'_stable' if job.get('stable') else '',int(job['ratio']*100))
    status_path=out/'status.json'; state=json.loads(status_path.read_text()) if status_path.exists() else {}
    def write():
        tmp=out/'status.tmp'; tmp.write_text(json.dumps(state,indent=2)); tmp.replace(status_path)
    while True:
        changed=False
        for job in jobs:
            key=job_key(job)
            trainer='train_pc_vlg_stable.py' if job.get('stable') else 'train_pc_vlg.py'
            if state.get(key,{}).get('status') in ['complete','failed']: continue
            config=root/'configs'/'pc_vlg'/((job['dataset']+'_stable'+str(int(job['ratio']*100)) if job.get('stable') else job['dataset'])+'.json')
            cfg=json.loads(config.read_text())
            reasons=[k+': '+str(cfg[k]) for k in ['manifest','medsam'] if not Path(cfg[k]).is_file()]
            if not Path(cfg['text_bank']).is_file() and not (Path(cfg['pubmed_bert'])/'DOWNLOAD_COMPLETE.json').is_file(): reasons.append('PubMed BERT download incomplete')
            if reasons:
                new=dict(status='waiting_inputs',missing=reasons)
                if state.get(key)!=new: print(key,json.dumps(new),flush=True); state[key]=new; changed=True
                continue
            try:
                used=int(subprocess.check_output(['nvidia-smi','-i',a.gpu,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
            except Exception as e:
                state[key]=dict(status='waiting_gpu',error=str(e)); changed=True; continue
            if used>256:
                state[key]=dict(status='waiting_gpu',used_mib=used); changed=True; continue
            if not Path(cfg['text_bank']).is_file():
                state[key]=dict(status='preparing_text'); write()
                prep=[sys.executable,'-u','prepare_pc_vlg_text.py','--knowledge',cfg['knowledge'],'--pubmed-bert',cfg['pubmed_bert'],'--out',str(Path(cfg['text_bank']).parent),'--gpu',a.gpu,'--local-only']
                with (out/(job['dataset']+'_text.log')).open('a') as log: code=subprocess.call(prep,stdout=log,stderr=subprocess.STDOUT)
                if code:
                    state[key]=dict(status='failed',stage='text',returncode=code); write(); continue
            smoke=root/'runs'/((key if job.get('stable') else job['dataset'])+'_preflight')
            if not (smoke/'PASSED').exists():
                smoke.mkdir(exist_ok=True)
                state[key]=dict(status='model_preflight'); write()
                probe=[sys.executable,'-u',trainer,'--config',str(config),'--gpu',a.gpu,'--ratio',str(job['ratio']),'--out',str(smoke),'--smoke-steps','2']
                with (smoke/'preflight.log').open('w') as log: code=subprocess.call(probe,stdout=log,stderr=subprocess.STDOUT)
                if code:
                    state[key]=dict(status='failed',stage='model_preflight',returncode=code); write(); continue
                (smoke/'PASSED').touch()
            run=root/'runs'/key
            run.mkdir(exist_ok=True)
            cmd=[sys.executable,'-u',trainer,'--config',str(config),'--gpu',a.gpu,'--ratio',str(job['ratio']),'--out',str(run)]
            if (run/'last.pt').exists(): cmd+=['--resume',str(run/'last.pt')]
            state[key]=dict(status='running',started=time.strftime('%F %T'),command=cmd); write()
            print('START',key,flush=True)
            with (run/'train.log').open('a') as log: code=subprocess.call(cmd,stdout=log,stderr=subprocess.STDOUT)
            state[key].update(status='complete' if code==0 else 'failed',returncode=code,finished=time.strftime('%F %T')); write()
            print('END',key,code,flush=True); changed=True
        if changed: write()
        if a.once or all(state.get(job_key(j),{}).get('status') in ['complete','failed'] for j in jobs): break
        time.sleep(a.poll)

if __name__=='__main__': main()
