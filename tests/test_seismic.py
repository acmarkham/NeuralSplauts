"""Dense-array partitions and propagation assumptions, independent of field labels."""

import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from splauts.config import Config
from splauts.model.generator import Splauts
from splauts.model.renderer import render
from splauts.training.mixtures import MixtureObservations, propose_events

try:
    from splauts.seismic_data import select_stations
    from splauts.seismic_experiments import interpolate_stations, station_nmse
    HAS_REAL = True
except ImportError:
    HAS_REAL = False


@unittest.skipUnless(HAS_REAL, "Install neural-splauts[real]")
class SeismicTests(unittest.TestCase):
    def test_missing_wav_tracks_excluded_and_spatial_splits_are_disjoint_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"array.csv"
            with path.open("w",newline="") as handle:
                writer=csv.writer(handle)
                writer.writerow(["TrackIndex","StationID","Lat","Lon"])
                for k in range(49):
                    writer.writerow([k+1,f"S{k}",.4560107+(k//7-3)*.0004,36.8429853+(k%7-3)*.0004])
            table,splits,mapping=select_stations(path,45,500,16,8,2026)
            second,again,_=select_stations(path,45,500,16,8,2026)
            self.assertEqual(len(table.ids),45)
            self.assertEqual(mapping["metadata_tracks_absent_from_wav"],[46,47,48,49])
            joined=np.concatenate(list(splits.values()))
            np.testing.assert_array_equal(np.sort(joined),np.arange(45))
            for key in splits:
                np.testing.assert_array_equal(splits[key],again[key])
            self.assertEqual(len(splits["test"]),21)

    def test_spreading_exponent_renderer_matches_analytic_power_ratios(self):
        splauts=Splauts(*[torch.tensor([[v]]) for v in [.5,25.,.15,2.,1.]])
        stations=torch.tensor([[100.,0.],[200.,0.]])
        times=torch.arange(100)/50
        frequencies=torch.tensor([23.,25.,27.])
        for exponent in (1.,2.):
            config=Config(sample_rate=500,n_fft=500,hop_length=10,sound_speed=330.,
                          power_spreading_exponent=exponent,distance_offset=25.)
            prediction=render(splauts,torch.tensor([[0.,0.]]),torch.tensor([0.]),torch.tensor([1.]),stations,times,frequencies,config)
            # Compare integrated power; propagation delays remain entirely inside the interval.
            ratio=float(prediction[0].sum()/prediction[1].sum())
            self.assertAlmostEqual(ratio,(225/125)**exponent,places=4)
        for exponent in (0.,-1.,float("nan")):
            with self.assertRaises(ValueError):
                Config(power_spreading_exponent=exponent)

    def test_dense_interpolation_uses_only_provided_fit_channels(self):
        fit=torch.tensor([[[1.,2.],[3.,4.]],[[5.,6.],[7.,8.]],[[9.,10.],[11.,12.]]])
        xy=torch.tensor([[0.,0.],[100.,0.],[0.,100.]])
        result=interpolate_stations(fit,xy,torch.tensor([[30.,30.],[40.,40.]]))
        self.assertEqual(result.shape,(2,2,2))
        self.assertTrue(bool((result>=fit.min()).all() and (result<=fit.max()).all()))
        np.testing.assert_allclose(station_nmse(result,result).numpy(),0)

    def test_surface_spreading_proposals_recover_exact_low_frequency_field(self):
        torch.set_num_threads(1)
        config=Config(sample_rate=500,n_fft=500,hop_length=50,scene_duration=12.,call_duration=3.,
                      area_size=600.,sound_speed=330.,distance_offset=25.,power_spreading_exponent=1.)
        stations=torch.tensor([[0.,0.],[600.,0.],[0.,600.],[600.,600.],[300.,0.],[0.,300.]])
        position=torch.tensor([[250.,350.]]) # 6x6 proposal cell centre
        fields=Splauts(*[torch.tensor([[v]]) for v in [1.5,26.,.5,2.,1.]])
        times=torch.arange(121)/10
        frequencies=torch.arange(8.,61.,2.)
        power=render(fields,position,torch.tensor([2.]),torch.tensor([100.]),stations,times,frequencies,config)
        proposal=propose_events(MixtureObservations(power,stations,times,frequencies),config,1,6,52.)
        self.assertLess(float(torch.linalg.vector_norm(proposal.positions-position)),80.)

    def test_window_proposals_ignore_a_louder_event_in_padding(self):
        torch.set_num_threads(1)
        config=Config(sample_rate=500,n_fft=500,hop_length=50,scene_duration=18.,call_duration=3.,
                      area_size=600.,sound_speed=330.,distance_offset=25.,power_spreading_exponent=1.)
        stations=torch.tensor([[0.,0.],[600.,0.],[0.,600.],[600.,600.]])
        position=torch.tensor([[250.,350.]])
        fields=Splauts(*[torch.tensor([[v]]) for v in [1.5,26.,.5,2.,1.]])
        times=torch.arange(181)/10
        frequencies=torch.arange(8.,61.,2.)
        def received(onset,amplitude):
            return render(fields,position,torch.tensor([onset]),torch.tensor([amplitude]),stations,times,frequencies,config)
        power=received(2.,100.)+received(11.,10000.)
        observed=MixtureObservations(power,stations,times,frequencies)
        ordinary=propose_events(observed,config,1,6,52.)
        masked=propose_events(observed,config,1,6,52.,(times>=2)&(times<8))
        self.assertGreater(float(ordinary.onsets[0]),9.)
        self.assertLess(float(masked.onsets[0]),5.)


if __name__=="__main__":
    unittest.main()
