from dataclasses import replace
import math
import unittest

import torch

from splauts.config import Config, station_layout
from splauts.model.call_prior import CallPriorGenerator, codes_from_logits, logits_from_codes
from splauts.model.generator import Splauts
from splauts.model.renderer import render
from splauts.propagation import absorption_transmission
from splauts.simulator.calls import CallSpec, instantaneous_frequency, moo_phase, power_spectrogram, synthesize_call
from splauts.simulator.scene import fractional_delay, random_scene
from splauts.training.frozen import fit_frozen_prior, interpolate_absorbed_bank
from splauts.training.mixtures import MixtureObservations, normalized_splauts
from splauts.training.pretrain import FrozenCallPrior, decoder_fingerprint, encode_calls, pretrain_call_prior


class AttenuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.set_flush_denormal(True)

    def test_moo_phase_derivative_and_rise_plateau_fall(self):
        spec = CallSpec("moo", duration=0.6, frequency=540., bandwidth=120., harmonic_count=4)
        t = torch.tensor([0., 0.075, 0.15, 0.3, 0.45, 0.525, 0.6], dtype=torch.float64, requires_grad=True)
        f = instantaneous_frequency(spec, t)
        torch.testing.assert_close(f, torch.tensor([420., 480., 540., 540., 540., 480., 420.], dtype=t.dtype))
        derivative, = torch.autograd.grad(moo_phase(spec, t).sum(), t)
        torch.testing.assert_close(derivative / (2*math.pi), f)
        wave = synthesize_call(spec)
        self.assertTrue(torch.isfinite(wave).all())
        field, times, frequencies = power_spectrogram(wave, Config())
        for time in (0.075, 0.3, 0.525):
            index = int((times-time).abs().argmin())
            fundamental = float(instantaneous_frequency(spec, times[index:index+1])[0])
            for order in range(1, 5):
                band = (frequencies - fundamental * order).abs() < 120
                peak = frequencies[band][field[index, band].argmax()]
                self.assertLess(abs(float(peak) - fundamental*order), 32.)

    def test_db_convention_frequency_and_distance_scaling(self):
        config = Config(absorption_db_per_km=10.)
        distance = torch.tensor([100., 200.])
        frequency = torch.tensor([0., 1000., 2000.])
        amplitude = absorption_transmission(distance, frequency, config)
        power = absorption_transmission(distance, frequency, config, power=True)
        expected_db = torch.tensor([[0., 1., 4.], [0., 2., 8.]])
        torch.testing.assert_close(-10 * power.log10(), expected_db, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(amplitude.square(), power)
        torch.testing.assert_close(absorption_transmission(distance, frequency, Config()), torch.ones(2, 3))
        for kwargs in ({"absorption_db_per_km": -1.}, {"absorption_power": 0.}, {"absorption_db_per_km": float("nan")}):
            with self.assertRaises(ValueError):
                Config(**kwargs)

    def test_waveform_propagation_matches_analytic_transmission(self):
        config = Config(absorption_db_per_km=10.)
        t = torch.arange(8000) / 8000
        for frequency in (500., 2000.):
            wave = torch.sin(2*math.pi*frequency*t)
            a = fractional_delay(wave, 0.3, 16000, 8000)
            b = fractional_delay(wave, 0.3, 16000, 8000, distance=200., config=config)
            ratio = (b[4400:8400].square().sum() / a[4400:8400].square().sum()).sqrt()
            expected = absorption_transmission(torch.tensor(200.), torch.tensor([frequency]), config)[0]
            torch.testing.assert_close(ratio, expected, atol=2e-4, rtol=2e-4)

    def test_renderer_spectral_transfer_and_position_gradient(self):
        config = Config(absorption_db_per_km=12.)
        atom = Splauts(*[torch.tensor([[v]], dtype=torch.float64) for v in (0.2, 1400., 0.07, 50., 1.)])
        xy = torch.tensor([[120., 140.]], dtype=torch.float64, requires_grad=True)
        stations = station_layout().double()
        times, frequencies = torch.arange(241).double()*0.01, torch.arange(257).double()*15.625
        def prediction(position, setting):
            return render(atom, position, torch.tensor([0.2]), torch.tensor([1.]), stations, times, frequencies, setting)
        plain = prediction(xy, replace(config, absorption_db_per_km=0.))
        attenuated = prediction(xy, config)
        distance = torch.cdist(stations, xy)[:, 0]
        transfer = absorption_transmission(distance, frequencies, config, power=True)
        torch.testing.assert_close(attenuated, plain * transfer[:, None])
        objective = (attenuated * times[None, :, None]).sum()
        gradient, = torch.autograd.grad(objective, xy)
        epsilon = 1e-3
        for axis in range(2):
            step = torch.zeros_like(xy); step[0, axis] = epsilon
            numeric = (((prediction(xy.detach()+step, config)-prediction(xy.detach()-step, config))
                        * times[None, :, None]).sum() / (2*epsilon))
            torch.testing.assert_close(gradient[0, axis], numeric, atol=1e-9, rtol=1e-4)

    def test_distance_time_interpolation(self):
        d, m, t, b = torch.meshgrid(torch.arange(3.), torch.arange(2.), torch.arange(5.), torch.arange(2.), indexing="ij")
        correlations = 3*d + m + 2*t + b
        energies = torch.arange(3.)[:, None] + torch.arange(2.)[None]
        distance = torch.tensor([[0.25, 1.5], [1., 2.]])
        query = torch.tensor([[[0.2, 1.5], [2.5, 3.]], [[1., 2.], [0., 3.2]]])
        aligned, energy = interpolate_absorbed_bank(correlations, energies, distance, query, 1., 1.)
        expected = 3*distance[:, :, None, None] + torch.arange(2.)[None, :, None, None] + 2*query[..., None] + torch.arange(2.)
        torch.testing.assert_close(aligned, expected)
        torch.testing.assert_close(energy, distance[:, :, None] + torch.arange(2.))

    def test_four_family_codes_and_supervised_moo_learning(self):
        config = Config(call_families=("moo", "tonal", "sweep", "pulse"), harmonic_count=4,
                        frequency_scale=0.3, splauts=32)
        calls = [CallSpec(kind, frequency=450., bandwidth=90., harmonic_count=4) for kind in config.call_families]
        codes = encode_calls(calls, config.latent_dim, config)
        torch.testing.assert_close(codes[:, :4], torch.eye(4))
        restored = codes_from_logits(torch.arange(4), logits_from_codes(codes, config), config.latent_dim, config)
        torch.testing.assert_close(restored, codes)
        prior = CallPriorGenerator(config).eval().requires_grad_(False)
        atoms = prior(codes.requires_grad_(), torch.full((4,), 0.4))
        self.assertEqual(atoms.frequency.shape, (4, 32))
        atoms.frequency.sum().backward()
        self.assertGreater(float(codes.grad[:, 4].abs().sum()), 1.)
        tiny = replace(config, call_families=("moo",))
        learned = pretrain_call_prior(tiny, per_family=4, steps=150, batch_size=4, frequency_step=200., log_every=0)
        self.assertLess(sum(learned.train_nmse)/4, learned.training_history[0]["shape_nmse"]*0.5)

    def test_absorbed_template_search_recovers_an_exact_grid_source(self):
        config = Config(area_size=60., harmonic_count=4, frequency_scale=0.3, splauts=16,
                        call_families=("moo",), absorption_db_per_km=10.)
        generator = CallPriorGenerator(config).eval().requires_grad_(False)
        code = encode_calls([CallSpec("moo", frequency=450., bandwidth=90., harmonic_count=4)], 16, config)
        relative = torch.arange(41)*0.01
        frequencies = torch.arange(257)*15.625
        prior = FrozenCallPrior(generator, code, relative, frequencies, decoder_fingerprint(generator))
        atoms = normalized_splauts(generator, code, torch.tensor([0.4]), relative, frequencies).detach()
        stations = station_layout(area_size=60.)
        times = torch.arange(161)*0.01
        power = render(atoms, torch.tensor([[30., 30.]]), torch.tensor([0.2]), torch.tensor([1.]),
                       stations, times, frequencies, config)
        result = fit_frozen_prior(MixtureObservations(power, stations, times, frequencies), prior, config,
                                  1, steps=0, proposal_steps=0, proposal_grid_size=3, log_every=0)
        torch.testing.assert_close(result.state.positions, torch.tensor([[30., 30.]]), atol=1e-4, rtol=0)
        self.assertLess(abs(float(result.state.onsets[0])-0.2), 0.02)
        self.assertEqual(result.decoder_fingerprint, prior.fingerprint)

    def test_attenuation_leaves_emissions_and_geometry_unchanged(self):
        config = Config(call_families=("moo", "tonal", "sweep", "pulse"), harmonic_count=4, frequency_scale=0.3)
        a = random_scene(config, 4, 4)
        b = random_scene(replace(config, absorption_db_per_km=10.), 4, 4)
        for x, y in zip(a.source_waveforms, b.source_waveforms):
            torch.testing.assert_close(x, y, atol=0, rtol=0)
        torch.testing.assert_close(a.delays, b.delays, atol=0, rtol=0)
        self.assertLess(float(b.waveforms.square().sum()), float(a.waveforms.square().sum()))
