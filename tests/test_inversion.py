import unittest

import torch

from splauts.config import Config, station_layout
from splauts.model.events import EventState
from splauts.model.generator import SplautGenerator
from splauts.model.renderer import render
from splauts.simulator.calls import CallSpec
from splauts.simulator.scene import SourceEvent, simulate_scene
from splauts.training.localize import initialize_positions, localize, normalized_mse, spatial_loss_grid
from splauts.training.train_calls import train_calls


class InversionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.config = Config()
        cls.calls = [CallSpec("tonal", frequency=1400), CallSpec("sweep", frequency=2100)]
        cls.training = train_calls(cls.calls, cls.config, steps=450)

    def test_call_learning_reduces_power_mse(self):
        losses = self.training.losses
        self.assertLess(losses[-1], losses[0] * 0.25)
        self.assertTrue(all(torch.isfinite(torch.tensor(losses))))

    def test_unsupervised_position_recovery_from_exact_renderer(self):
        config = self.config
        generator = SplautGenerator(config)
        latents = torch.zeros(1, config.latent_dim)
        durations = torch.tensor([0.4])
        splauts = generator(latents, durations).detach()
        positions = torch.tensor([[120., 155.]])
        onsets = torch.tensor([0.2])
        stations = station_layout()
        times, frequencies = torch.linspace(0, 2, 101), torch.linspace(0, 4000, 65)
        observation = render(splauts, positions, onsets, torch.ones(1), stations, times, frequencies, config)
        initial = EventState(torch.tensor([[80., 200.]]), onsets, torch.ones(1), durations, latents)
        # Do not even pass true_positions: labels cannot enter the objective.
        result = localize(generator, initial, observation, stations, times, frequencies, config, steps=220)
        self.assertLess(float((result.state.positions - positions).norm()), 0.1)
        self.assertLess(result.reconstruction_losses[-1], result.reconstruction_losses[0] * 1e-4)
        self.assertTrue(all(p.grad is None for p in generator.parameters()))

    def test_waveform_single_and_overlapping_known_events(self):
        config, training = self.config, self.training
        stations = station_layout()
        events = [SourceEvent(torch.tensor([120., 155.]), 0.2, self.calls[0]),
                  SourceEvent(torch.tensor([200., 115.]), 0.27, self.calls[1])]
        for count in (1, 2):
            scene = simulate_scene(events[:count], stations, config)
            positions = torch.stack([e.position for e in scene.events])
            state = EventState(torch.tensor([[70., 220.], [230., 75.]])[:count],
                               torch.tensor([e.onset for e in scene.events]), torch.ones(count),
                               training.durations[:count], training.latents[:count])
            result = localize(training.generator, state, scene.power, stations, scene.times,
                              scene.frequencies, config, steps=250)
            self.assertLess(float((result.state.positions - positions).norm(dim=-1).max()), 8.0)
            self.assertLess(result.reconstruction_losses[-1], result.reconstruction_losses[0] * 0.5)

    def test_optional_joint_refinement_and_label_weights(self):
        training, config = self.training, self.config
        stations = station_layout()
        times, frequencies = torch.linspace(0, 2, 81), training.frequencies[::4]
        truth = EventState(torch.tensor([[120., 155.]]), torch.tensor([0.2]), torch.ones(1),
                           training.durations[:1], training.latents[:1])
        observation = render(training.generator(truth.latents, truth.durations).detach(), truth.positions,
                             truth.onsets, truth.amplitudes, stations, times, frequencies, config)
        initial = truth.detach()
        initial.positions += torch.tensor([[10., -10.]])
        initial.onsets += 0.01
        initial.amplitudes *= 0.9
        result = localize(training.generator, initial, observation, stations, times, frequencies, config,
                          steps=180, optimize_onsets=True, optimize_amplitudes=True, optimize_latents=True)
        self.assertLess(result.reconstruction_losses[-1], result.reconstruction_losses[0] * 0.1)
        self.assertTrue(torch.isfinite(result.state.latents).all())
        baseline = localize(training.generator, initial, observation, stations, times, frequencies, config, steps=3)
        reporting = localize(training.generator, initial, observation, stations, times, frequencies, config,
                             steps=3, true_positions=truth.positions + 50)
        torch.testing.assert_close(baseline.state.positions, reporting.state.positions, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            localize(training.generator, initial, observation, stations, times, frequencies, config, position_weight=1)
        axis, losses = spatial_loss_grid(training.generator, truth, observation, stations,
                                         times, frequencies, config, resolution=5)
        self.assertEqual((len(axis), losses.shape), (5, (5, 5)))
        self.assertTrue(torch.isfinite(losses).all())

    def test_grid_initialization_uses_reconstruction_without_labels(self):
        training, config = self.training, self.config
        stations = station_layout()
        times, frequencies = torch.linspace(0, 2, 81), training.frequencies[::4]
        truth = EventState(torch.tensor([[120., 155.]]), torch.tensor([0.2]), torch.ones(1),
                           training.durations[:1], training.latents[:1])
        splauts = training.generator(truth.latents, truth.durations).detach()
        def predict(state):
            return render(splauts, state.positions, state.onsets, state.amplitudes,
                          stations, times, frequencies, config)
        observation = predict(truth)
        initial = truth.detach()
        initial.positions = torch.tensor([[70., 220.]])
        coarse = initialize_positions(training.generator, initial, observation, stations,
                                      times, frequencies, config, resolution=7)
        self.assertLess(float(normalized_mse(predict(coarse), observation)),
                        float(normalized_mse(predict(initial), observation)))
        self.assertLess(float((coarse.positions - truth.positions).norm()),
                        float((initial.positions - truth.positions).norm()))
        torch.testing.assert_close(initial.positions, torch.tensor([[70., 220.]]))


if __name__ == "__main__":
    unittest.main()
