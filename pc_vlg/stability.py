"""Opt-in ACDC stability improvements, separate from the paper configuration."""
import copy,math
import torch
from torch import nn
from torch.nn import functional as F


class CoarseEMA(nn.Module):
    def __init__(self,student,decay=0.99):
        super().__init__(); self.model=copy.deepcopy(student).eval().requires_grad_(False); self.decay=decay
    @torch.no_grad()
    def update(self,student):
        for target,source in zip(self.model.parameters(),student.parameters()): target.lerp_(source.detach(),1-self.decay)
        for target,source in zip(self.model.buffers(),student.buffers()): target.copy_(source)
        self.model.eval()


class ValidationGuard:
    def __init__(self,patience=8,min_delta=0.001,min_steps=2000,smoothing_window=1):
        self.patience=patience; self.min_delta=min_delta; self.min_steps=min_steps
        self.smoothing_window=max(1,int(smoothing_window)); self.history=[]
        self.best=-float('inf'); self.patience_best=-float('inf'); self.best_step=0; self.stale=0
    def update(self,score,step):
        if not math.isfinite(score): raise FloatingPointError('Invalid validation score')
        self.history.append(float(score))
        smooth=sum(self.history[-self.smoothing_window:])/len(self.history[-self.smoothing_window:])
        improved=smooth>self.patience_best+self.min_delta
        # A new absolute best is always saved; patience resets only for material improvement.
        absolute_best=score>self.best
        if improved: self.stale=0; self.patience_best=smooth
        else: self.stale+=1
        if absolute_best: self.best=score; self.best_step=step
        return absolute_best,step>=self.min_steps and self.stale>=self.patience


def foreground_dice(prob,target,mask=None):
    k=prob.shape[1]; valid=(target>=0)&(target<k)
    if mask is not None: valid=valid&mask
    gt=F.one_hot(target.clamp(0,k-1),k).permute(0,3,1,2).float()[:,1:]*valid[:,None]
    p=prob.float()[:,1:]*valid[:,None]
    axes=(0,2,3)
    return 1-((2*(p*gt).sum(axes)+1e-5)/(p.square().sum(axes)+gt.square().sum(axes)+1e-5)).mean()


def confident_consistency(logits,teacher_prob,agreement_prob=None,fg_threshold=0.85,bg_threshold=0.98):
    teacher_prob=teacher_prob.detach().float()
    confidence,target=teacher_prob.max(1)
    threshold=torch.where(target==0,confidence.new_tensor(bg_threshold),confidence.new_tensor(fg_threshold))
    valid=confidence>=threshold
    if agreement_prob is not None:
        ac,ay=agreement_prob.detach().float().max(1)
        valid=valid&(ay==target)&(ac>=threshold)
    fg=valid&(target!=0)
    if int(fg.sum())<16:
        return logits.sum()*0,0.0
    per_pixel=F.cross_entropy(logits.float(),target,reduction='none')
    pieces=[per_pixel[valid&(target==c)].mean() for c in range(logits.shape[1]) if bool((valid&(target==c)).any())]
    ce=torch.stack(pieces).mean()  # class-balanced, not dominated by easy background pixels
    dice=foreground_dice(logits.float().softmax(1),target,valid)
    return 0.5*ce+dice,float(valid.float().mean())


def stable_loss(coarse,refined,teacher_prob,labels,labeled,step,cfg):
    weights=coarse.new_tensor([cfg.get('background_weight',0.25)]+[1.]*(coarse.shape[1]-1)).float()
    c=coarse[labeled].float(); r=refined[labeled].float(); y=labels[labeled]
    sup=F.cross_entropy(c,y,weight=weights)+F.cross_entropy(r,y,weight=weights)
    sup=sup+foreground_dice(c.softmax(1),y)+foreground_dice(r.softmax(1),y)
    unsup=coarse.sum()*0; coverage=0.
    start=cfg.get('consistency_start',1000); warm=cfg.get('warmup',5000)
    mu=0. if step<start else cfg.get('mu_max',0.02)*math.exp(-5*(1-min((step-start)/max(1,warm),1))**2)
    if (~labeled).any() and mu>0:
        cu=coarse[~labeled]; ru=refined[~labeled]; tp=teacher_prob[~labeled]
        a,ca=confident_consistency(cu,tp,ru.detach().float().softmax(1),cfg.get('fg_threshold',0.85),cfg.get('bg_threshold',0.98))
        b,cb=confident_consistency(ru,tp,None,cfg.get('fg_threshold',0.85),cfg.get('bg_threshold',0.98))
        unsup=a+b; coverage=(ca+cb)/2
    return sup+mu*unsup,dict(supervised=sup.detach(),consistency=unsup.detach(),mu=mu,coverage=coverage)
