"""Trainable interpretation of equations (12)-(14); literal paper option retained."""
import math
import torch
from torch.nn import functional as F


def dice_loss(prob,target,classes):
    valid=(target>=0)&(target<classes)
    safe=target.clamp(0,classes-1)
    gt=F.one_hot(safe,classes).permute(0,3,1,2).float()*valid[:,None]
    pred=prob.float()*valid[:,None]
    axes=(0,2,3)
    dice=(2*(pred*gt).sum(axes)+1e-5)/(pred.square().sum(axes)+gt.square().sum(axes)+1e-5)
    return 1-dice.mean()


def progressive_loss(coarse,refined,labels,labeled,step,total=90000,mu_max=0.1,warmup=10000,literal=False):
    classes=coarse.shape[1]
    if not labeled.any(): raise ValueError('each batch must have labeled data')
    lc=coarse[labeled].float(); lr=refined[labeled].float(); y=labels[labeled]
    dc=F.one_hot(lc.argmax(1),classes).permute(0,3,1,2).float() if literal else lc.softmax(1)
    dr=F.one_hot(lr.argmax(1),classes).permute(0,3,1,2).float() if literal else lr.softmax(1)
    sup=F.cross_entropy(lc,y)+F.cross_entropy(lr,y)+dice_loss(dc,y,classes)+dice_loss(dr,y,classes)
    unsup=coarse.sum()*0
    if (~labeled).any():
        cu=coarse[~labeled].float(); ru=refined[~labeled].float()
        unsup=dice_loss(cu.softmax(1),ru.detach().argmax(1),classes)+dice_loss(ru.softmax(1),cu.detach().argmax(1),classes)
    mu=mu_max*math.exp(-5*(1-min(max(step,0)/max(warmup,1),1))**2)
    return sup+mu*unsup,dict(supervised=sup.detach(),consistency=unsup.detach(),mu=mu)


def paired_contrastive(z,zprime,tau=0.07,literal=False):
    z=F.normalize(z.float(),dim=-1); zprime=F.normalize(zprime.float(),dim=-1)
    if z.shape[0]<2: raise ValueError('contrastive batch requires at least two distinct concepts')
    logits=z @ zprime.T / tau
    target=torch.arange(z.shape[0],device=z.device)
    if literal:
        negative=logits.masked_fill(torch.eye(len(z),device=z.device,dtype=torch.bool),float('-inf'))
        return (negative.logsumexp(1)+negative.logsumexp(0)-2*logits.diag()).mean()
    return F.cross_entropy(logits,target)+F.cross_entropy(logits.T,target)
