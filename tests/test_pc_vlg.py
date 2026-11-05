import unittest,torch
from torch import nn
from pc_vlg.graph import PositionCorrelatedGraph
from pc_vlg.prompts import ConsistencyAwarePromptGenerator
from pc_vlg.losses import progressive_loss,paired_contrastive
from pc_vlg.model import VLGRNet


class GraphTests(unittest.TestCase):
    def setUp(self): torch.manual_seed(31); torch.set_num_threads(2)
    def test_graph_probability_and_gradients(self):
        m=PositionCorrelatedGraph(32,3)
        x=torch.randn(2,32,8,8,requires_grad=True); q=torch.randn(2,4,32,requires_grad=True)
        y,t=m(x,q,torch.zeros(4,4,dtype=torch.long),torch.tensor([0.5,0.4]))
        self.assertTrue(torch.allclose(t['alpha'].sum(-1),torch.ones(2,4),atol=1e-6))
        self.assertEqual(float(t['edges'].diagonal(dim1=-2,dim2=-1).abs().sum()),0)
        self.assertTrue(bool((t['edges'].sum(-1)<=1+1e-6).all()))
        y.square().mean().backward()
        self.assertTrue(bool(torch.isfinite(x.grad).all() and torch.isfinite(q.grad).all()))
    def test_permutation_equivariance(self):
        m=PositionCorrelatedGraph(32,3).eval(); x=torch.randn(1,32,8,8); q=torch.randn(1,4,32)
        r=torch.randint(0,3,(4,4)); p=torch.tensor([2,0,3,1]); mid=torch.tensor([0.5])
        a,_=m(x,q,r,mid); b,_=m(x,q[:,p],r[p][:,p],mid)
        self.assertTrue(torch.allclose(a,b,atol=1e-5))
    def test_uniform_attention_closes_edges(self):
        m=PositionCorrelatedGraph(32,3); x=torch.randn(1,32,8,8); q=torch.zeros(1,3,32)
        _,t=m(x,q,torch.zeros(3,3,dtype=torch.long),torch.tensor([0.5]))
        self.assertLess(float(t['edges'].abs().max()),1e-10)
    def test_canonical_midline_translation(self):
        m=PositionCorrelatedGraph(32,3); x=torch.randn(1,32,8,8); q=torch.randn(1,3,32); r=torch.zeros(3,3,dtype=torch.long)
        a,_=m(x,q,r,torch.tensor([0.5]))
        transform=torch.tensor([[[1.,0,0.2],[0,1,0]]])
        b,_=m(x,q,r,torch.tensor([0.6]),transform)
        self.assertTrue(torch.allclose(a,b,atol=2e-5))
    def test_unlabeled_prompt_no_gt_leak(self):
        logits=torch.randn(2,4,16,16); labeled=torch.tensor([True,False]); y=torch.randint(0,4,(2,16,16))
        f=ConsistencyAwarePromptGenerator(); a,av=f(logits,y,labeled)
        y[1]=999; b,bv=f(logits,y,labeled)
        self.assertTrue(torch.equal(a,b) and torch.equal(av,bv))
    def test_supervised_and_consistency_gradients(self):
        c=torch.randn(4,4,8,8,requires_grad=True); r=torch.randn_like(c,requires_grad=True)
        y=torch.randint(0,4,(4,8,8)); y[2:]=-1; labeled=torch.tensor([1,1,0,0],dtype=torch.bool)
        loss,parts=progressive_loss(c,r,y,labeled,50000); loss.backward()
        self.assertGreater(float(c.grad[:2].abs().sum()),0); self.assertGreater(float(c.grad[2:].abs().sum()),0)
        self.assertTrue(bool(torch.isfinite(r.grad).all())); self.assertAlmostEqual(parts['mu'],0.1)
    def test_text_pair_contrastive(self):
        z=torch.eye(4,requires_grad=True); loss=paired_contrastive(z,z); loss.backward()
        self.assertTrue(bool(torch.isfinite(z.grad).all()))
    def test_full_four_stage_six_query_backward(self):
        class Prior(nn.Module):
            def __init__(self): super().__init__(); self.proj=nn.Conv2d(1,256,1); self.requires_grad_(False)
            def forward(self,image,boxes,valid):
                with torch.no_grad(): e=self.proj(torch.nn.functional.adaptive_avg_pool2d(image,(8,8)))
                return e,e.mean((2,3))[:,None].expand(-1,6,-1)
        model=VLGRNet(Prior(),torch.randn(3,32),torch.zeros(3,3,dtype=torch.long),4,sizes=(8,8,16,16))
        image=torch.rand(2,1,32,32); coarse=torch.randn(2,4,32,32,requires_grad=True)
        label=torch.randint(0,4,(2,32,32)); labeled=torch.tensor([True,False]); label[1]=-1
        result,traces=model(image,coarse,torch.tensor([0.5,0.5]),label,labeled,return_graph=True)
        self.assertEqual(len(traces),4); self.assertEqual(len(model.query_blocks),6)
        loss,_=progressive_loss(coarse,result,label,labeled,1000); loss.backward()
        self.assertTrue(all(p.grad is None for p in model.prior.parameters()))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for block in model.query_blocks for p in block.parameters()))
        self.assertFalse(model.prior.training)

if __name__=='__main__': unittest.main(verbosity=2)
