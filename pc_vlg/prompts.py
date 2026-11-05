"""Class-wise CAPG implementation choice; no labels may enter unlabeled prompts."""
import torch
from torch import nn


class ConsistencyAwarePromptGenerator(nn.Module):
    def __init__(self, confidence=0.6, agreement=0.5, padding=0.05):
        super().__init__()
        self.confidence,self.agreement,self.padding=confidence,agreement,padding

    @torch.no_grad()
    def forward(self, logits, labels=None, labeled=None):
        b,k,h,w=logits.shape
        prob=logits.detach().float().softmax(1)
        confidence,pred=prob.max(1)
        boxes=torch.zeros(b,k-1,4,device=logits.device)
        valid=torch.zeros(b,k-1,device=logits.device,dtype=torch.bool)
        if labeled is None: labeled=torch.zeros(b,device=logits.device,dtype=torch.bool)
        for i in range(b):
            for c in range(1,k):
                pm=(pred[i]==c)&(confidence[i]>=self.confidence)
                if bool(labeled[i]):
                    if labels is None: raise ValueError('labeled prompts require labels')
                    gt=labels[i]==c
                    union=(gt|pm).sum()
                    iou=(gt&pm).sum().float()/union.clamp_min(1)
                    region=(gt|pm) if iou>=self.agreement else gt
                else:
                    region=pm  # never inspect unlabeled ground truth
                ys,xs=torch.where(region)
                if xs.numel():
                    x0,x1=xs.min().float(),xs.max().float()+1
                    y0,y1=ys.min().float(),ys.max().float()+1
                    dx=(x1-x0)*self.padding; dy=(y1-y0)*self.padding
                    boxes[i,c-1]=torch.stack(((x0-dx).clamp(0,w),(y0-dy).clamp(0,h),(x1+dx).clamp(0,w),(y1+dy).clamp(0,h)))
                    valid[i,c-1]=True
        return boxes,valid
