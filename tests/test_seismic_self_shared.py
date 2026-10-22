import unittest

import numpy as np
import torch

from splauts.seismic_self_shared import assess


class SeismicSelfSharedMetricsTests(unittest.TestCase):
    def test_stationary_power_improves_raw_error_without_explaining_calls(self):
        times = torch.arange(40)/40
        call = torch.sin(2*torch.pi*times)[:,None] * torch.tensor([[.3,.1]])
        target = (1+call)[None].expand(4,-1,-1)
        constant = torch.ones_like(target)
        splits = {"test": np.arange(4)}
        metrics,_ = assess(constant,target,constant,splits)
        m = metrics["test"]
        self.assertLess(m["raw_nmse"]["mean"], .1)
        self.assertAlmostEqual(m["dynamic_nmse"]["mean"], 1.)
        self.assertEqual(m["dynamic_correlation"]["mean"], 0.)

    def test_correct_dynamics_score_perfectly_despite_stationary_self_power(self):
        times = torch.arange(40)/40
        call = torch.exp(-((times-.4)/.1).square())[:,None]*torch.tensor([[1.,.5]])
        prediction = call[None].expand(4,-1,-1)
        target = prediction+2.
        splits = {"test": np.arange(4)}
        metrics,_ = assess(prediction,target,torch.zeros_like(target),splits)
        m = metrics["test"]
        self.assertLess(m["dynamic_nmse"]["mean"], 1e-10)
        self.assertAlmostEqual(m["dynamic_correlation"]["mean"], 1., places=6)
        self.assertGreater(m["raw_nmse"]["mean"], .5)


if __name__ == "__main__":
    unittest.main()
