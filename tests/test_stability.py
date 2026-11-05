import unittest,torch
from pc_vlg.stability import CoarseEMA,ValidationGuard,confident_consistency,stable_loss

class StabilityTests(unittest.TestCase):
    def test_ema_is_frozen_and_updates(self):
        m=torch.nn.Linear(2,2,bias=False); e=CoarseEMA(m,0.5); before=e.model.weight.detach().clone()
        with torch.no_grad(): m.weight.add_(2)
        e.update(m); self.assertTrue(torch.allclose(e.model.weight,before+1)); self.assertFalse(e.model.weight.requires_grad)
    def test_low_confidence_and_background_skip(self):
        logits=torch.randn(1,4,8,8,requires_grad=True)
        for prob in [torch.full_like(logits,0.25),torch.nn.functional.one_hot(torch.zeros(1,8,8,dtype=torch.long),4).permute(0,3,1,2).float()]:
            loss,cov=confident_consistency(logits,prob); self.assertEqual(float(loss),0); self.assertEqual(cov,0)
    def test_teacher_gradient_detached(self):
        logits=torch.randn(1,4,8,8,requires_grad=True); t=torch.full_like(logits,-8.,requires_grad=True)
        with torch.no_grad(): t[:,1]=8
        loss,cov=confident_consistency(logits,t.softmax(1)); loss.backward()
        self.assertGreater(cov,0.9); self.assertIsNone(t.grad); self.assertGreater(float(logits.grad.abs().sum()),0)
    def test_guard_keeps_best_and_stops(self):
        g=ValidationGuard(patience=3,min_delta=0.001,min_steps=100)
        self.assertEqual(g.update(0.86,0),(True,False))
        g.update(0.82,100); g.update(0.81,200)
        self.assertEqual(g.update(0.7,300),(False,True)); self.assertEqual(g.best,0.86); self.assertEqual(g.best_step,0)
    def test_small_cumulative_improvements_reset_patience(self):
        g=ValidationGuard(patience=3,min_steps=0); g.update(0.80,0); g.update(0.8006,100)
        improved,stop=g.update(0.8012,200); self.assertTrue(improved); self.assertFalse(stop); self.assertEqual(g.stale,0)
    def test_warmup_supervised_gradient(self):
        c=torch.randn(4,4,8,8,requires_grad=True); r=torch.randn_like(c,requires_grad=True)
        y=torch.randint(0,4,(4,8,8)); labeled=torch.tensor([1,1,0,0],dtype=torch.bool); y[2:]=-1
        loss,info=stable_loss(c,r,c.detach().softmax(1),y,labeled,0,{})
        loss.backward(); self.assertEqual(info['mu'],0); self.assertTrue(torch.isfinite(c.grad).all()); self.assertTrue(torch.isfinite(r.grad).all())

if __name__=='__main__': unittest.main(verbosity=2)
