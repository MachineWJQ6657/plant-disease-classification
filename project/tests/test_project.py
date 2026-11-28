"""Run from project/: python -m unittest discover -s tests -v"""
import csv
import random
import shutil
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from analysis import audit
from data import canonical_path, disk_path
from train import crop_parameters, deterministic, fresh_transform

RESULTS = Path(__file__).resolve().parents[1] / 'results'


class EvidenceTests(unittest.TestCase):
    def test_saved_results_pass_audit(self):
        for seed in [664, 665, 666]:
            self.assertEqual(audit(RESULTS / f'seed_{seed}', RESULTS / 'manifest.csv')['status'], 'PASS')

    def assert_rejected(self, filename, change):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            for p in [*(RESULTS / 'seed_664').glob('*.csv'), RESULTS / 'manifest.csv']:
                shutil.copy2(p, out / p.name)
            with (out / filename).open(encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                fields, rows = reader.fieldnames, list(reader)
            change(rows)
            with (out / filename).open('w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaises(ValueError):
                audit(out, out / 'manifest.csv')

    def test_missing_prediction_is_rejected(self):
        self.assert_rejected('predictions.csv', lambda rows: rows.pop())

    def test_corrupt_probability_is_rejected(self):
        self.assert_rejected('predictions.csv', lambda rows: rows[0].update(p_0='nan'))

    def test_cross_split_duplicate_is_rejected(self):
        def change(rows):
            next(r for r in rows if r['split'] == 'pv_train')['sha256'] = next(
                r for r in rows if r['split'] == 'pv_test')['sha256']
        self.assert_rejected('manifest.csv', change)

    def test_wrong_saved_accuracy_is_rejected(self):
        self.assert_rejected('overall.csv', lambda rows: rows[0].update(accuracy='1.0'))

    def test_windows_invalid_upstream_name_roundtrips(self):
        original = 'PlantDoc/train/Tomato leaf/photo%20.jpeg?itok=abc.jpg'
        physical = disk_path(Path('data'), original)
        self.assertNotIn('?', physical.name)
        self.assertEqual(canonical_path(physical, Path('data')), original)
        self.assertNotEqual(physical, disk_path(Path('data'), original.replace('?', '%3F')))


class FreshViewTests(unittest.TestCase):
    rgb = np.random.default_rng(664).integers(0, 256, (128, 128, 3), dtype=np.uint8)

    def test_views_are_replayable_and_change_each_epoch_and_seed(self):
        first = fresh_transform(self.rgb, 'strong', 664, 0, 17)
        self.assertTrue(torch.equal(first, fresh_transform(self.rgb, 'strong', 664, 0, 17)))
        self.assertFalse(torch.equal(first, fresh_transform(self.rgb, 'strong', 664, 1, 17)))
        self.assertFalse(torch.equal(first, fresh_transform(self.rgb, 'strong', 665, 0, 17)))

    def test_thread_scheduling_does_not_change_views(self):
        expected = [fresh_transform(self.rgb, 'strong', 664, 2, i) for i in range(8)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            actual = list(pool.map(lambda i: fresh_transform(self.rgb, 'strong', 664, 2, i), range(8)))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(expected, actual)))

    def test_crops_are_in_bounds_and_views_finite(self):
        for i in range(50):
            top, left, height, width = crop_parameters(random.Random(i), 128, 128)
            self.assertTrue(0 <= top < top + height <= 128 and 0 <= left < left + width <= 128)
        for condition in ['ordinary', 'strong']:
            view = fresh_transform(self.rgb, condition, 664, 0, 0)
            self.assertEqual(tuple(view.shape), (3, 128, 128))
            self.assertTrue(torch.isfinite(view).all())

    def test_deterministic_settings(self):
        saved = (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.benchmark,
                 torch.backends.cudnn.deterministic, torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        try:
            torch.backends.cudnn.allow_tf32 = True
            deterministic()
            self.assertTrue(torch.are_deterministic_algorithms_enabled())
            self.assertFalse(torch.backends.cudnn.benchmark or torch.backends.cuda.matmul.allow_tf32
                             or torch.backends.cudnn.allow_tf32)
        finally:
            torch.use_deterministic_algorithms(saved[0])
            (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
             torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) = saved[1:]


if __name__ == '__main__':
    unittest.main()
