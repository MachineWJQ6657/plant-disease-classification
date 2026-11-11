"""Regression tests for fresh, reproducible, thread-independent training views."""
from concurrent.futures import ThreadPoolExecutor
import random
import unittest
import numpy as np
import torch
from experiment_fresh import crop_parameters, fresh_transform
from evaluate import configure_replay


class FreshViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.rgb=np.random.default_rng(664).integers(0,256,(128,128,3),dtype=np.uint8)

    def test_fresh_views_change_and_are_replayable(self):
        first=fresh_transform(self.rgb,'strong',664,0,17)
        replay=fresh_transform(self.rgb,'strong',664,0,17)
        next_epoch=fresh_transform(self.rgb,'strong',664,1,17)
        next_seed=fresh_transform(self.rgb,'strong',665,0,17)
        self.assertTrue(torch.equal(first,replay))
        self.assertFalse(torch.equal(first,next_epoch))
        self.assertFalse(torch.equal(first,next_seed))

    def test_thread_scheduling_does_not_change_views(self):
        expected=[fresh_transform(self.rgb,'strong',664,2,i) for i in range(8)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            actual=list(pool.map(lambda i:fresh_transform(self.rgb,'strong',664,2,i),range(8)))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(expected,actual)))

    def test_crop_is_in_bounds_and_views_are_finite(self):
        for i in range(50):
            top,left,height,width=crop_parameters(random.Random(i),128,128)
            self.assertTrue(0<=top<top+height<=128)
            self.assertTrue(0<=left<left+width<=128)
        for condition in ['ordinary','strong']:
            view=fresh_transform(self.rgb,condition,664,0,0)
            self.assertEqual(tuple(view.shape),(3,128,128))
            self.assertTrue(torch.isfinite(view).all())

    def test_replay_restores_training_precision(self):
        previous=(torch.are_deterministic_algorithms_enabled(),torch.backends.cudnn.benchmark,
            torch.backends.cudnn.deterministic,torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32)
        try:
            torch.backends.cudnn.allow_tf32=True
            settings=configure_replay('fresh_views_v1')
            self.assertFalse(settings['cudnn_tf32'])
            self.assertFalse(settings['matmul_tf32'])
            self.assertTrue(settings['deterministic_algorithms'])
            self.assertFalse(settings['cudnn_benchmark'])
        finally:
            torch.use_deterministic_algorithms(previous[0])
            torch.backends.cudnn.benchmark=previous[1]
            torch.backends.cudnn.deterministic=previous[2]
            torch.backends.cuda.matmul.allow_tf32=previous[3]
            torch.backends.cudnn.allow_tf32=previous[4]


if __name__=='__main__':
    unittest.main()
