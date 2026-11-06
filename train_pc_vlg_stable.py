"""PC-VLG segmentation training: 90k iterations, batch four, separate optimizers."""
import argparse,json,os,random,time,math
from pathlib import Path
import numpy as np
import torch
from pc_vlg.data import audit_manifest,SliceDataset,TwoStreamBatchSampler
from pc_vlg.model import FrozenMedSAM,VLGRNet,build_coarse
from pc_vlg.stability import CoarseEMA,ValidationGuard,stable_loss
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
    init=cfg.get('init_checkpoint')
    if init and not a.resume:
        state=torch.load(init,map_location='cpu')
        if state.get('config',{}).get('labeled_cases')!=sorted(labeled_cases): raise ValueError('Initialization labeled-patient split mismatch')
        coarse.load_state_dict(state['coarse'],strict=True)
        missing,unexpected=refined.load_state_dict(state['refined'],strict=False)
        permitted=('alignment_logit_scale','graph_residual_scale')
        if unexpected or any(not key.endswith(permitted) for key in missing):
            raise RuntimeError('Unexpected graph initialization mismatch: missing={} unexpected={}'.format(missing,unexpected))
    ema=CoarseEMA(coarse,cfg.get('ema_decay',0.99))
    guard=ValidationGuard(cfg.get('patience',8),cfg.get('min_delta',0.001),cfg.get('min_steps',2000),cfg.get('smoothing_window',1))
    plateau_multiplier=1.0
    lr_coarse=cfg.get('lr_coarse',0.003); lr_refined=cfg.get('lr_refined',1e-5)
    optc=torch.optim.SGD(coarse.parameters(),lr=lr_coarse,momentum=0.9,weight_decay=1e-4)
    optr=torch.optim.AdamW((p for p in refined.parameters() if p.requires_grad),lr=lr_refined,weight_decay=0.01)
    amp=bool(cfg.get('amp',False)); scaler=torch.cuda.amp.GradScaler(enabled=amp,init_scale=128)
    step=0; best=-1.; consecutive_nonfinite=0
    if a.resume:
        state=torch.load(a.resume,map_location='cpu')
        coarse.load_state_dict(state['coarse']); refined.load_state_dict(state['refined'])
        optc.load_state_dict(state['optimizer_coarse']); optr.load_state_dict(state['optimizer_refined']); scaler.load_state_dict(state['scaler'])
        step=state['step']; best=state['best_val_dsc']
        ema.model.load_state_dict(state['ema']); guard.__dict__.update(state['guard']); plateau_multiplier=state['plateau_multiplier']
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
            optimizer_refined=optr.state_dict(),scaler=scaler.state_dict(),step=step,best_val_dsc=best,config=run,ema=ema.model.state_dict(),guard=guard.__dict__,plateau_multiplier=plateau_multiplier,
            rng_python=random.getstate(),rng_numpy=np.random.get_state(),rng_torch=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all()),path)
    if not a.resume and not a.smoke_steps:
        baseline=evaluate(ema.model,manifest['val'],classes,device,surface_metrics=False)
        baseline.update(step=0,split='val',weights='ema',baseline=True)
        guard.update(baseline['dsc'],0); best=baseline['dsc']
        with (out/'val.jsonl').open('a') as f: f.write(json.dumps(baseline)+'\n')
        save(out/'best.pt')
        atomic_save(dict(coarse=ema.model.state_dict(),classes=classes,step=0,val_dsc=best,weights='ema'),out/'best_coarse.pt')
        print('BASELINE_VAL '+str(best),flush=True)
    coarse.train(); refined.train()
    while step<total:
        try: batch=next(loader_iter)
        except StopIteration: loader_iter=iter(loader); batch=next(loader_iter)
        image=batch['image'].to(device); label=batch['label'].to(device)
        labeled=batch['labeled'].to(device); midline=batch['midline'].to(device); affine=batch['affine'].to(device)
        # Explicitly mask unlabeled labels even if a custom dataset supplies hidden GT.
        label=label.clone(); label[~labeled]=-1
        optc.zero_grad(set_to_none=True); optr.zero_grad(set_to_none=True)
        warm=min(1.,(step+1)/cfg.get('lr_warmup',500))
        factor=(0.1+0.9*0.5*(1+math.cos(math.pi*step/total)))*warm*plateau_multiplier
        optc.param_groups[0]['lr']=lr_coarse*factor; optr.param_groups[0]['lr']=lr_refined*factor
        with torch.no_grad(),torch.cuda.amp.autocast(enabled=amp):
            teacher_prob=ema.model(image).float().softmax(1)
        if cfg.get('photometric',True):
            image=(image*torch.empty(len(image),1,1,1,device=device).uniform_(0.9,1.1)+torch.randn_like(image)*0.02).clamp(0,1)
        with torch.cuda.amp.autocast(enabled=amp):
            mc=coarse(image)
            if isinstance(mc,(tuple,list)): mc=mc[0]
            mr=refined(image,mc,midline,label,labeled,coordinate_transform=affine)
            loss,parts=stable_loss(mc,mr,teacher_prob,label,labeled,step,cfg)
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
        scaler.step(optc); scaler.step(optr); scaler.update(); ema.update(coarse); step+=1
        if step%20==0 or step==1:
            rec=dict(step=step,loss=float(loss.detach()),supervised=float(parts['supervised']),consistency=float(parts['consistency']),mu=parts['mu'],pseudo_coverage=parts['coverage'],lr_coarse=optc.param_groups[0]['lr'],elapsed=time.monotonic()-start)
            print(json.dumps(rec),flush=True)
            with (out/'train.jsonl').open('a') as f: f.write(json.dumps(rec)+'\n')
        if a.smoke_steps and step>=a.smoke_steps:
            print('REAL_MODEL_PREFLIGHT_OK step {} peak_cuda_mib {:.1f}'.format(step,torch.cuda.max_memory_allocated()/1024**2),flush=True)
            return
        if step%cfg.get('val_every',1000)==0 or step==total:
            # Save before evaluation so a failed metric pass does not lose weights.
            save(out/'last.pt')
            val=evaluate(ema.model,manifest['val'],classes,device,surface_metrics=False)
            val['step']=step; val['split']='val'; val['weights']='ema'
            new_best,stop=guard.update(val['dsc'],step)
            with (out/'val.jsonl').open('a') as f: f.write(json.dumps(val)+'\n')
            print('VAL '+json.dumps({k:v for k,v in val.items() if k!='per_case'}),flush=True)
            if cfg.get('test_every_validation',False):
                test=evaluate(ema.model,manifest['test'],classes,device,surface_metrics=False)
                test.update(step=step,split='test_monitor',weights='ema',checkpoint_selection='validation_only')
                with (out/'test_monitor.jsonl').open('a') as f: f.write(json.dumps(test)+'\n')
                print('TEST_MONITOR '+json.dumps({k:v for k,v in test.items() if k!='per_case'}),flush=True)
            if new_best:
                best=val['dsc']; save(out/'best.pt')
                atomic_save(dict(coarse=ema.model.state_dict(),classes=classes,step=step,val_dsc=best,weights='ema'),out/'best_coarse.pt')
            if guard.stale and guard.stale%3==0:
                plateau_multiplier=max(0.05,plateau_multiplier*0.5)
                print('PLATEAU_LR_MULTIPLIER '+str(plateau_multiplier),flush=True)
            save(out/'last.pt')
            if stop:
                record=dict(reason='validation_patience',step=step,best_step=guard.best_step,best_val_dsc=guard.best,patience=guard.patience)
                (out/'early_stop.json').write_text(json.dumps(record,indent=2)); print('EARLY_STOP '+json.dumps(record),flush=True)
                break
        elif step%cfg.get('save_every',1000)==0: save(out/'last.pt')
    (out/'validation_result.json').write_text(json.dumps(dict(best_val_dsc=best,best_step=guard.best_step,completed_step=step,target=0.85,target_met=best>0.85,selected_on='val_only'),indent=2))
    if not cfg.get('run_test',False): return
    best_state=torch.load(out/'best_coarse.pt',map_location='cpu'); coarse.load_state_dict(best_state['coarse'])
    test=evaluate(coarse,manifest['test'],classes,device)
    test['split']='test'; test['selected_on']='val_dsc'; test['checkpoint_step']=best_state['step']
    (out/'test.json').write_text(json.dumps(test,indent=2)); print('TEST '+json.dumps(test),flush=True)

if __name__=='__main__': main()
