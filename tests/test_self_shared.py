import unittest
from unittest.mock import patch

import torch

from splauts.config import Config, station_layout
from splauts.model.generator import Splauts
from splauts.model.events import EventState
from splauts.model.renderer import render
from splauts.training.mixtures import MixtureObservations
from splauts.training.self_shared import (SelfSharedLoss, coherent_stations,
                                          decomposition_loss, fit_self_shared, received_components)


class SelfSharedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.set_flush_denormal(True)

    def test_self_fills_underprediction_but_cannot_cancel_overprediction(self):
        y = torch.ones(4, 11, 7)
        weights = SelfSharedLoss()
        support = torch.tensor([4.])
        zero, local, terms = decomposition_loss(torch.zeros(1,4,11,7),y,torch.zeros(1),support,weights)
        torch.testing.assert_close(local,y,rtol=0,atol=0)
        self.assertEqual(float(terms["reconstruction"]),0.)
        self.assertEqual(float(terms["self_energy"]),1.)
        good, local, terms = decomposition_loss(torch.full((1,4,11,7),.3),y,torch.tensor([.3]),support,weights)
        torch.testing.assert_close(local,torch.full_like(y,.7))
        self.assertLess(float(good),float(zero))
        bad, local, terms = decomposition_loss(torch.full((1,4,11,7),1.5),y,torch.tensor([1.5]),support,weights)
        self.assertEqual(float(local.sum()),0.)
        self.assertGreater(float(bad),float(zero))

    def test_stationary_shared_spectrum_is_not_evidence_of_a_propagating_event(self):
        weights = SelfSharedLoss()
        spectral = torch.tensor([1.,2.,1.,.1])[None,None,None].expand(2,8,41,-1)
        observation = spectral[0] * torch.arange(1.,9.)[:,None,None]
        self.assertTrue(bool((coherent_stations(spectral,observation,weights)<.1).all()))

    def test_spatial_cost_is_invariant_to_splitting_one_location_into_groups(self):
        y = torch.ones(4,11,7)
        weights = SelfSharedLoss()
        one = decomposition_loss(torch.full((1,4,11,7),.5),y,torch.tensor([.5]),
                                 torch.tensor([4.]),weights,torch.tensor([[10.,20.]]))[2]["spatial_sparsity"]
        two_components = torch.stack((torch.full_like(y,.2),torch.full_like(y,.3)))
        two = decomposition_loss(two_components,y,torch.tensor([.2,.3]),torch.tensor([4.,4.]),
                                 weights,torch.tensor([[10.,20.],[10.,20.]]))[2]["spatial_sparsity"]
        separate = decomposition_loss(two_components,y,torch.tensor([.2,.3]),torch.tensor([4.,4.]),
                                      weights,torch.tensor([[10.,20.],[210.,220.]]))[2]["spatial_sparsity"]
        torch.testing.assert_close(one,two)
        self.assertGreater(float(separate),float(two)*1.9)

    def test_one_sensor_transient_fails_the_group_support_requirement(self):
        weights = SelfSharedLoss()
        t = torch.linspace(-1,1,41)
        pattern = torch.exp(-t.square()/.03)[:,None]*torch.tensor([[1.,.5,.1]])
        template = pattern[None,None].expand(1,8,-1,-1)
        shared = pattern[None].expand(8,-1,-1).clone()
        coherent = coherent_stations(template,shared,weights)
        self.assertGreater(float(coherent),7.9)
        shared[1:] = 1.
        local = coherent_stations(template,shared,weights)
        self.assertLess(float(local),1.1)

    def test_core_loss_ignores_padding_but_retains_full_self_field(self):
        y = torch.ones(4, 11, 7)
        components = torch.full((1, 4, 11, 7), .3)
        mask = torch.zeros(11, dtype=torch.bool)
        mask[3:8] = True
        args = (torch.tensor([.3]), torch.tensor([4.]), SelfSharedLoss())
        expected = decomposition_loss(components[:, :, mask], y[:, mask], *args)[0]
        y[:, ~mask] = 1000.
        components[:, :, ~mask] = 2000.
        actual, local, _ = decomposition_loss(components, y, *args, time_mask=mask)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(local.shape, y.shape)
        self.assertEqual(float(local[:, ~mask].sum()), 0.)

    def test_core_normalization_and_support_ignore_padding(self):
        config = Config(sample_rate=500, n_fft=250, hop_length=25, call_duration=1.)
        stations = station_layout(4)
        times, frequencies = torch.arange(61)/20, torch.arange(20.,41.,2.)
        splauts = Splauts(*[torch.tensor([[v]]) for v in [.5,30.,.15,2.,1.]])
        state = EventState(torch.tensor([[150.,150.]]),torch.tensor([.4]),torch.ones(1),
                           torch.ones(1),torch.zeros(1,config.latent_dim))
        y = render(splauts,state.positions,state.onsets,state.amplitudes,stations,times,frequencies,config)+1e-7
        mask = (times>=.8)&(times<1.8)
        obs = MixtureObservations(y,stations,times,frequencies)
        expected = received_components(splauts,state,obs,config,torch.tensor([.2]),time_mask=mask)
        y[:,~mask] = 1e6
        actual = received_components(splauts,state,obs,config,torch.tensor([.2]),time_mask=mask)
        for a,b in zip(actual,expected):
            torch.testing.assert_close(a,b)

    def test_mixture_only_birth_recovers_a_shared_source_and_leaves_local_bursts(self):
        config = Config(sample_rate=500,n_fft=250,hop_length=25,scene_duration=4.,call_duration=1.,
                        area_size=300.,distance_offset=25.,sound_speed=330.,splauts=20)
        stations = station_layout(8,"irregular")
        times, frequencies = torch.arange(81)/20, torch.arange(16.,61.,2.)
        source = Splauts(*[torch.tensor([[v]]) for v in [.5,30.,.13,2.,1.]])
        truth = torch.tensor([[150.,150.]])
        clean = render(source,truth,torch.tensor([.7]),torch.tensor([5000.]),stations,times,frequencies,config)
        local = torch.full_like(clean,.0001)
        for m in range(8):
            local[m] += .035*torch.exp(-((times[:,None]-(2.+m*.1))/.12).square()
                                      -((frequencies[None]-(18.+m*4.))/1.5).square())
        observed = MixtureObservations(clean+local,stations,times,frequencies)
        with patch("splauts.simulator.calls.synthesize_call",side_effect=AssertionError("No clean calls")):
            fit = fit_self_shared(observed,config,capacity=3,warmup_steps=100,steps=220,
                                  geometry_steps=40,grid_size=7,log_every=0)
        self.assertEqual(fit.history[0]["gate_mass"],[0.,0.,0.])
        self.assertEqual(fit.history[0]["reconstruction"],0.)
        active = fit.gates>.03
        self.assertTrue(bool(active.any()))
        self.assertLess(float(torch.cdist(fit.state.positions[active],truth).min()),15.)
        error = (fit.shared-clean).square().sum()/clean.square().sum()
        self.assertLess(float(error),.4)
        # The unrelated bursts must substantially remain in the station-local field.
        mask = local>.02
        self.assertGreater(float(fit.self_power[mask].sum()/local[mask].sum()),.6)


if __name__ == "__main__":
    unittest.main()
