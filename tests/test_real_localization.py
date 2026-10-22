"""Verify the independent waveform baseline's travel-delay sign and geometry."""

import unittest
import numpy as np

try:
    from scipy import signal
    from splauts.real_tdoa import spatial_correlation
    HAS_REAL = True
except ImportError:
    HAS_REAL = False


@unittest.skipUnless(HAS_REAL, "Install neural-splauts[real]")
class RealLocalizationTests(unittest.TestCase):
    def test_waveform_delays_localize_and_station_permutation_preserves_map(self):
        rate = 512
        stations = np.array([[0,0], [300,0], [300,300], [0,300], [140,350], [-40,100.]])
        source = np.array([85.,125.])
        t = np.arange(6*rate)/rate
        clean = signal.sosfiltfilt(signal.butter(4,[20,110],btype="bandpass",fs=rate,output="sos"),
                                  np.random.default_rng(3).normal(size=len(t)))
        clean *= (t>1)&(t<4)
        distances = np.linalg.norm(stations-source,axis=1)
        audio = np.stack([np.interp(t-d/343,t,clean,left=0,right=0)/(d+50) for d in distances])
        candidates = np.array([source, [150,150], [20,250], [270,30]])
        values, diagnostics = spatial_correlation(audio,stations,candidates)
        self.assertEqual(int(values.argmax()),0)
        self.assertGreater(values[0],.7)
        permutation = np.array([3,5,1,0,4,2])
        permuted, _ = spatial_correlation(audio[permutation],stations[permutation],candidates)
        np.testing.assert_allclose(values,permuted,atol=1e-12)
        self.assertEqual(len(diagnostics),15)


if __name__ == "__main__":
    unittest.main()
