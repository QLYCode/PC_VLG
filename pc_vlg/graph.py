"""Position-correlated graph operations, paper equations (6)-(11)."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def positional_encoding(xy, dim):
    if dim % 4:
        raise ValueError('position dimension must be divisible by four')
    freq = torch.exp(-math.log(10000.0) * torch.arange(dim//4, device=xy.device, dtype=xy.dtype) / max(1, dim//4-1))
    phase = xy.unsqueeze(-1) * freq * (2*math.pi)
    return torch.cat((phase.sin(), phase.cos()), -1).flatten(-2)


class PositionCorrelatedGraph(nn.Module):
    def __init__(self, dim=256, relations=16):
        super().__init__()
        for name in ['vq','vs','vv','vp','vt','eq','ek','message','out','back']:
            setattr(self,name,nn.Linear(dim,dim,bias=False))
        self.geometry = nn.Sequential(nn.Linear(3,dim),nn.ReLU(),nn.Linear(dim,1,bias=False))
        self.relations = nn.Embedding(relations,256)
        self.relation_score = nn.Linear(256,1,bias=False)
        self.ffn = nn.Sequential(nn.Linear(dim,dim*4),nn.GELU(),nn.Linear(dim*4,dim))
        # Sharpen query-to-pixel localization gradually; start neutral for stability.
        self.alignment_logit_scale = nn.Parameter(torch.zeros(()))
        # Keep graph messages a small residual early in training to limit noisy propagation.
        self.graph_residual_scale = nn.Parameter(torch.tensor(0.2))

    def forward(self, feature, query, relation_ids, midline, coordinate_transform=None):
        b,d,h,w=feature.shape
        if query.shape[1]<2 or h*w<2:
            raise ValueError('graph requires >=2 concepts and >=2 spatial positions')
        # FP32 attention statistics prevent low-precision entropy/softmax overflow.
        with torch.autocast(device_type=feature.device.type, enabled=False):
            s=feature.float().flatten(2).transpose(1,2); q=query.float()
            align_scale=self.alignment_logit_scale.exp().clamp(0.5,2.0)
            alpha=((self.vq(q) @ self.vs(s).transpose(1,2))*align_scale).softmax(-1)
            visual=alpha @ s
            yy,xx=torch.meshgrid(torch.linspace(0,1,h,device=s.device),torch.linspace(0,1,w,device=s.device),indexing='ij')
            xy=torch.stack((xx,yy),-1).reshape(h*w,2)
            if coordinate_transform is not None:
                canonical=(2*xy-1)[None] @ coordinate_transform[:,:,:2].float().transpose(1,2)+coordinate_transform[:,:,2].float()[:,None]
                canonical=(canonical+1)/2
                centroid=alpha @ canonical
            else:
                centroid=alpha @ xy
            centered=torch.stack((centroid[...,0]-midline.float().view(b,1),centroid[...,1]),-1)
            nodes=self.vv(visual)+self.vp(positional_encoding(centered,d))+self.vt(q)
            delta=centered[:,:,None]-centered[:,None,:]
            geo=torch.cat((delta,delta.norm(dim=-1,keepdim=True)),-1)
            logits=self.eq(nodes) @ self.ek(nodes).transpose(1,2)
            logits=logits+self.geometry(geo).squeeze(-1)+self.relation_score(self.relations(relation_ids)).squeeze(-1)
            mask=torch.eye(q.shape[1],device=q.device,dtype=torch.bool)
            logits=logits.masked_fill(mask,float('-inf'))
            entropy=-(alpha*alpha.clamp_min(1e-12).log()).sum(-1)/math.log(h*w)
            certainty=(1-entropy).clamp(0,1)
            gate=certainty[:,:,None]*certainty[:,None,:]
            edges=logits.softmax(-1)*gate  # gate AFTER softmax; do not renormalize
            residual_scale=self.graph_residual_scale.clamp(0.05,0.5)
            nodes=self.ffn(nodes+residual_scale*(edges @ self.message(nodes)))
            beta=(self.back(nodes) @ s.transpose(1,2)).softmax(1)
            refined=s+residual_scale*(beta.transpose(1,2) @ self.out(nodes))
        result=refined.transpose(1,2).reshape(b,d,h,w).to(feature.dtype)
        return result,dict(alpha=alpha,centroids=centroid,entropy=entropy,edges=edges,nodes=nodes)
