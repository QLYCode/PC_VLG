import tempfile,unittest,zipfile,json,hashlib
from pathlib import Path
from unittest.mock import patch
import bootstrap_pc_vlg_data as bootstrap
from download_flare21 import verify

class FlareTests(unittest.TestCase):
    def test_checksum_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x'; p.write_bytes(b'known')
            verify(p,hashlib.md5(b'known').hexdigest())
            with self.assertRaises(ValueError): verify(p,'0'*32)
    def test_zip_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.zip'
            with zipfile.ZipFile(p,'w') as z: z.writestr('../escape.txt','bad')
            with self.assertRaises(ValueError): bootstrap.safe_extract(p,Path(d)/'extract')
    def test_official_pairing_and_case_split(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/'TRAINING_VERIFIED').touch()
            for archive,folder in [('TrainingImg.zip','TrainingImg'),('TrainingMask.zip','TrainingMask')]:
                with zipfile.ZipFile(root/archive,'w') as z:
                    for i in range(361):
                        suffix='_0000' if folder=='TrainingImg' else ''
                        z.writestr(folder+'/train_%03d'%i+suffix+'.nii.gz',b'fixture')
            with patch.object(bootstrap,'run_prepare') as prepare:
                bootstrap.flare(root)
                dataset,splits=prepare.call_args.args
                self.assertEqual(dataset,'flare21')
                self.assertEqual([len(splits[s]) for s in ['train','val','test']],[260,26,75])
                ids=[r['case_id'] for split in splits.values() for r in split]
                self.assertEqual(len(set(ids)),361)
                self.assertFalse(any('ValidationImg' in r['image'] for split in splits.values() for r in split))

if __name__=='__main__': unittest.main(verbosity=2)
