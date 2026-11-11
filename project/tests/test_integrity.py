"""Scientific integrity checks must reject corrupted evidence."""
import csv
from pathlib import Path
import shutil
import tempfile
import unittest
from verify import audit
from paths import disk_path, canonical_path

ROOT=Path(__file__).resolve().parents[1]

class IntegrityTests(unittest.TestCase):
    def test_windows_invalid_upstream_name_roundtrips(self):
        root=Path('data')
        original='PlantDoc/train/Tomato leaf/photo%20.jpeg?itok=abc.jpg'
        physical=disk_path(root,original)
        self.assertNotIn('?',physical.name)
        self.assertEqual(canonical_path(physical,root),original)
        self.assertNotEqual(physical,disk_path(root,original.replace('?', '%3F')))

    def test_saved_experiment_passes(self):
        self.assertEqual(audit(ROOT/'results')['status'],'PASS')

    def corrupt(self,filename,change):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)
            for p in (ROOT/'results').glob('*.csv'):
                shutil.copy2(p,out/p.name)
            p=out/filename
            with p.open(encoding='utf-8',newline='') as f:
                reader=csv.DictReader(f); fields=reader.fieldnames; rows=list(reader)
            change(rows)
            with p.open('w',encoding='utf-8',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
            with self.assertRaises(ValueError):
                audit(out)

    def test_missing_test_prediction_fails(self):
        self.corrupt('predictions.csv',lambda rows:rows.pop())

    def test_corrupt_probabilities_fail(self):
        self.corrupt('predictions.csv',lambda rows:rows[0].update(p_0='nan'))

    def test_cross_split_duplicate_fails(self):
        def change(rows):
            train=next(r for r in rows if r['split']=='pv_train')
            test=next(r for r in rows if r['split']=='pv_test')
            train['sha256']=test['sha256']
        self.corrupt('manifest.csv',change)

    def test_wrong_saved_accuracy_fails(self):
        self.corrupt('overall.csv',lambda rows:rows[0].update(accuracy='1.0'))

if __name__=='__main__':
    unittest.main()
