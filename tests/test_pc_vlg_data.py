import json,tempfile,unittest
from pathlib import Path
import numpy as np
from pc_vlg.data import audit_manifest,SliceDataset,TwoStreamBatchSampler

class DataTests(unittest.TestCase):
    def test_patient_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'m.json'
            p.write_text(json.dumps(dict(train=[dict(case_id='p1')],val=[dict(case_id='p1')],test=[dict(case_id='p3')])))
            with self.assertRaises(ValueError): audit_manifest(p)
    def test_patient_label_counts_and_sampler(self):
        with tempfile.TemporaryDirectory() as d:
            rows=[dict(case_id='p%03d'%i,volume_id='phase%d'%j) for i in range(75) for j in [0,1]]
            p=Path(d)/'m.json'; p.write_text(json.dumps(dict(train=rows,val=[dict(case_id='v')],test=[dict(case_id='t')])) )
            _,labeled=audit_manifest(p,0.1); self.assertEqual(len(labeled),7)
            sampler=TwoStreamBatchSampler(rows,labeled)
            for batch in sampler:
                self.assertEqual(sum(rows[i]['case_id'] in labeled for i in batch),2)
    def test_unlabeled_mask_is_never_read(self):
        with tempfile.TemporaryDirectory() as d:
            image=Path(d)/'im.npy'; np.save(image,np.ones((1,8,8),dtype=np.float32))
            ds=SliceDataset([dict(case_id='unlabeled',image=str(image),label='DOES_NOT_EXIST',slice=0)],{'labeled'})
            r=ds[0]; self.assertFalse(r['labeled']); self.assertTrue(bool((r['label']==-1).all()))

if __name__=='__main__': unittest.main(verbosity=2)
