import unittest
from unittest.mock import patch

import torch

from splauts.config import Config, station_layout
from splauts.mixture_experiments import match_positions
from splauts.model.generator import Splauts, SplautGenerator, evaluate_field
from splauts.model.renderer import render
from splauts.simulator.scene import random_scene
from splauts.training.mixtures import MixtureObservations, fit_mixtures, mixture_losses, normalized_splauts


class MixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.set_flush_denormal(True)

    def test_learns_from_station_observations_without_source_call_access(self):
        config = Config(n_fft=256, hop_length=160, splauts=12, latent_dim=8)
        stations = station_layout()
        times, frequencies = torch.arange(121) * 0.02, torch.linspace(0, 4000, 129)
        splauts = Splauts(torch.tensor([[0.2], [0.2]]), torch.tensor([[900.], [2200.]]),
                          torch.full((2, 1), 0.05), torch.full((2, 1), 70.), torch.tensor([[2.], [1.5]]))
        positions = torch.tensor([[100., 140.], [210., 180.]])
        observation = render(splauts, positions, torch.tensor([0.2, 0.5]), torch.ones(2),
                             stations, times, frequencies, config)
        observed = MixtureObservations(observation, stations, times, frequencies)
        # No isolated calls even exist in this experiment. Guard both the waveform
        # synthesizer and supervised acoustic trainer against accidental access.
        with patch("splauts.simulator.calls.synthesize_call", side_effect=AssertionError("No source access")), \
             patch("splauts.training.train_calls.train_calls", side_effect=AssertionError("No isolated training")):
            result = fit_mixtures(observed, config, 2, warmup_steps=100, steps=120,
                                  geometry_steps=20, proposal_grid_size=7, log_every=0)
        final_power, _ = mixture_losses(result.prediction, observation)
        self.assertLess(float(final_power), result.power_losses[0] * 0.8)
        self.assertLess(float(final_power), 0.5)
        self.assertEqual(result.state.positions.shape, (2, 2))
        self.assertEqual(result.splauts.weight.shape, (2, 12))
        self.assertTrue(torch.isfinite(result.prediction).all())
        self.assertTrue((result.prediction >= 0).all())

    def test_hidden_scene_labels_cannot_change_fitting(self):
        config = Config(n_fft=256, hop_length=160, splauts=8, latent_dim=8)
        scene = random_scene(config, 2, 4)
        observed = MixtureObservations(scene.power, scene.stations, scene.times, scene.frequencies)
        options = dict(warmup_steps=8, steps=8, geometry_steps=2, proposal_grid_size=5, log_every=0)
        first = fit_mixtures(observed, config, 2, **options)
        for event in scene.events:
            event.position = torch.tensor([0., 0.])
            event.onset = 100.
            event.amplitude = 999.
        scene.source_power = [torch.full_like(p, float("nan")) for p in scene.source_power]
        scene.source_waveforms = []
        second = fit_mixtures(observed, config, 2, **options)
        torch.testing.assert_close(first.prediction, second.prediction, rtol=0, atol=0)
        torch.testing.assert_close(first.state.positions, second.state.positions, rtol=0, atol=0)

    def test_five_call_simulation_and_evaluation_assignment(self):
        scene = random_scene(Config(n_fft=256, hop_length=160), 5, 8)
        self.assertEqual(len(scene.events), 5)
        self.assertEqual(scene.power.shape[0], 8)
        estimated = torch.tensor([[200., 100.], [10., 20.], [80., 90.]])
        truth = estimated[torch.tensor([2, 0, 1])]
        self.assertEqual(match_positions(estimated, truth), [2, 0, 1])

    def test_field_normalization_removes_decoder_power_gauge(self):
        config = Config(splauts=8, latent_dim=8)
        generator = SplautGenerator(config)
        latents, durations = torch.zeros(3, 8), torch.full((3,), 0.4)
        times, frequencies = torch.arange(41) * 0.01, torch.linspace(0, 4000, 65)
        before = evaluate_field(normalized_splauts(generator, latents, durations, times, frequencies), times, frequencies)
        torch.testing.assert_close(before.square().mean((1, 2)).sqrt(), torch.ones(3))
        with torch.no_grad():
            generator.intensity_scale.mul_(10000)
        after = evaluate_field(normalized_splauts(generator, latents, durations, times, frequencies), times, frequencies)
        torch.testing.assert_close(after, before)

    def test_rejects_empty_energy_and_bad_geometry(self):
        config = Config()
        times, frequencies = torch.arange(10) * 0.01, torch.arange(10) * 100.
        observed = MixtureObservations(torch.zeros(4, 10, 10), station_layout(), times, frequencies)
        with self.assertRaises(ValueError):
            fit_mixtures(observed, config, 2, warmup_steps=0, steps=0)
        observed = MixtureObservations(torch.ones(3, 10, 10), station_layout()[:3], times, frequencies)
        with self.assertRaises(ValueError):
            observed.validate()


if __name__ == "__main__":
    unittest.main()
