"""PC-VLG segmentation training: 90k iterations, batch four, separate optimizers."""
import argparse,json,os,random,time
from pathlib import Path
import numpy as np
import torch
from pc_vlg.data import audit_manifest,SliceDataset,TwoStreamBatchSampler
from pc_vlg.model import FrozenMedSAM,VLGRNet,build_coarse
from pc_vlg.losses import progressive_loss
from pc_vlg.metrics import evaluate


def seed_worker(worker_id):
    seed=torch.initial_seed()%2**32
    np.random.seed(seed); random.seed(seed)


def atomic_save(payload,path):
    temporary=Path(str(path)+'.tmp')
    torch.save(payload,temporary); os.replace(temporary,path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True); p.add_argument('--gpu',default='3')
    p.add_argument('--ratio',type=float,choices=[0.1,0.2],default=0.1)
    p.add_argument('--smoke-steps',type=int,default=0)
    p.add_argument('--out',required=True); p.add_argument('--resume')
    a=p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
    cfg=json.loads(Path(a.config).read_text()); seed=int(cfg.get('seed',1337))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    device=torch.device('cuda')
    manifest,labeled_cases=audit_manifest(cfg['manifest'],a.ratio,seed)
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    if (out/'last.pt').exists() and not a.resume: raise ValueError('Existing run needs explicit --resume')
    classes=cfg['classes']; batch_size=cfg.get('batch_size',4)
    bank=torch.load(cfg['text_bank'],map_location='cpu')
    if len(bank['metadata']['concepts'])!=classes-1 or bank['metadata']['concepts']!=cfg['concepts']:
        raise ValueError('Text concepts/order must match segmentation labels 1..C')
    if bank['metadata']['steps']<=0: raise ValueError('Text encoder must undergo semantic contrastive adaptation')
    coarse=build_coarse(classes).to(device)
    refined=VLGRNet(FrozenMedSAM(cfg['medsam']),bank['embeddings'],bank['relation_ids'],classes,
                   relations=len(bank['metadata']['relations'])).to(device)
    optc=torch.optim.SGD(coarse.parameters(),lr=0.03,momentum=0.9,weight_decay=1e-4)
    optr=torch.optim.AdamW((p for p in refined.parameters() if p.requires_grad),lr=1e-4,weight_decay=0.01)
    amp=bool(cfg.get('amp',False)); scaler=torch.cuda.amp.GradScaler(enabled=amp,init_scale=128)
    step=0; best=-1.; consecutive_nonfinite=0
    if a.resume:
        state=torch.load(a.resume,map_location='cpu')
        coarse.load_state_dict(state['coarse']); refined.load_state_dict(state['refined'])
        optc.load_state_dict(state['optimizer_coarse']); optr.load_state_dict(state['optimizer_refined']); scaler.load_state_dict(state['scaler'])
        step=state['step']; best=state['best_val_dsc']
        random.setstate(state['rng_python']); np.random.set_state(state['rng_numpy']); torch.set_rng_state(state['rng_torch'])
        torch.cuda.set_rng_state_all(state['rng_cuda'])
    dataset=SliceDataset(manifest['train'],labeled_cases,augment=True)
    sampler=TwoStreamBatchSampler(manifest['train'],labeled_cases,batch_size,cfg.get('labeled_bs',2),seed)
    generator=torch.Generator().manual_seed(seed+step)
    loader=torch.utils.data.DataLoader(dataset,batch_sampler=sampler,num_workers=cfg.get('workers',4),worker_init_fn=seed_worker,generator=generator,pin_memory=True)
    loader_iter=iter(loader)
    run=dict(config=cfg,ratio=a.ratio,labeled_cases=sorted(labeled_cases),gpu=a.gpu,source='Paper-6229',text_metadata=bank['metadata'])
    (out/'run_config.json').write_text(json.dumps(run,indent=2))
    print('RUN '+json.dumps(run),flush=True)
    total=cfg.get('iterations',90000); start=time.monotonic()
    def save(path):
        atomic_save(dict(coarse=coarse.state_dict(),refined=refined.state_dict(),optimizer_coarse=optc.state_dict(),
            optimizer_refined=optr.state_dict(),scaler=scaler.state_dict(),step=step,best_val_dsc=best,config=run,
            rng_python=random.getstate(),rng_numpy=np.random.get_state(),rng_torch=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all()),path)
    coarse.train(); refined.train()
    while step<total:
        try: batch=next(loader_iter)
        except StopIteration: loader_iter=iter(loader); batch=next(loader_iter)
        image=batch['image'].to(device); label=batch['label'].to(device)
        labeled=batch['labeled'].to(device); midline=batch['midline'].to(device); affine=batch['affine'].to(device)
        # Explicitly mask unlabeled labels even if a custom dataset supplies hidden GT.
        label=label.clone(); label[~labeled]=-1
        optc.zero_grad(set_to_none=True); optr.zero_grad(set_to_none=True)
        factor=(1-step/total)**0.9
        optc.param_groups[0]['lr']=0.03*factor; optr.param_groups[0]['lr']=1e-4*factor
        with torch.cuda.amp.autocast(enabled=amp):
            mc=coarse(image)
            if isinstance(mc,(tuple,list)): mc=mc[0]
            mr=refined(image,mc,midline,label,labeled,coordinate_transform=affine)
            loss,parts=progressive_loss(mc,mr,label,labeled,step,total,warmup=cfg.get('warmup',10000),literal=cfg.get('literal_eq13',False))
        if not torch.isfinite(loss): raise FloatingPointError('Non-finite loss at step '+str(step))
        scaler.scale(loss).backward(); scaler.unscale_(optc); scaler.unscale_(optr)
        normc=torch.nn.utils.clip_grad_norm_(coarse.parameters(),1,error_if_nonfinite=False)
        normr=torch.nn.utils.clip_grad_norm_((p for p in refined.parameters() if p.requires_grad),1,error_if_nonfinite=False)
        if not bool(torch.isfinite(normc) and torch.isfinite(normr)):
            consecutive_nonfinite+=1
            if not amp or consecutive_nonfinite>20: raise FloatingPointError('Persistent nonfinite gradients')
            scaler.update(new_scale=max(scaler.get_scale()/2,1))
            print('AMP overflow: skip BOTH optimizers, reduce scale to '+str(scaler.get_scale()),flush=True)
            continue
        consecutive_nonfinite=0
        scaler.step(optc); scaler.step(optr); scaler.update(); step+=1
        if step%20==0 or step==1:
            rec=dict(step=step,loss=float(loss.detach()),supervised=float(parts['supervised']),consistency=float(parts['consistency']),mu=parts['mu'],elapsed=time.monotonic()-start)
            print(json.dumps(rec),flush=True)
            with (out/'train.jsonl').open('a') as f: f.write(json.dumps(rec)+'\n')
        if a.smoke_steps and step>=a.smoke_steps:
            print('REAL_MODEL_PREFLIGHT_OK step {} peak_cuda_mib {:.1f}'.format(step,torch.cuda.max_memory_allocated()/1024**2),flush=True)
            return
        if step%cfg.get('val_every',1000)==0 or step==total:
            val=evaluate(coarse,manifest['val'],classes,device)
            val['step']=step; val['split']='val'
            with (out/'val.jsonl').open('a') as f: f.write(json.dumps(val)+'\n')
            print('VAL '+json.dumps({k:v for k,v in val.items() if k!='per_case'}),flush=True)
            if val['dsc']>best:
                best=val['dsc']; save(out/'best.pt')
                atomic_save(dict(coarse=coarse.state_dict(),classes=classes,step=step,val_dsc=best),out/'best_coarse.pt')
            save(out/'last.pt')
        elif step%cfg.get('save_every',1000)==0: save(out/'last.pt')
    best_state=torch.load(out/'best_coarse.pt',map_location='cpu'); coarse.load_state_dict(best_state['coarse'])
    test=evaluate(coarse,manifest['test'],classes,device)
    test['split']='test'; test['selected_on']='val_dsc'; test['checkpoint_step']=best_state['step']
    (out/'test.json').write_text(json.dumps(test,indent=2)); print('TEST '+json.dumps(test),flush=True)

if __name__=='__main__': main()
