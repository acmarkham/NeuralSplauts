import unittest

import torch

from splauts.config import Config, station_layout
from splauts.model.generator import Splauts, SplautGenerator, evaluate_field
from splauts.model.renderer import render


class RendererTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_bounded_decoder_and_nonnegative_fields(self):
        config = Config()
        generator = SplautGenerator(config)
        durations = torch.tensor([0.4, 0.6])
        splauts = generator(torch.randn(2, config.latent_dim), durations)
        self.assertTrue(((splauts.time >= 0) & (splauts.time <= durations[:, None])).all())
        self.assertTrue(((splauts.frequency >= 0) & (splauts.frequency <= 4000)).all())
        for value in (splauts.time_width, splauts.frequency_width, splauts.weight):
            self.assertTrue((value > 0).all())
        self.assertTrue((splauts.time_width <= durations[:, None] / 3).all())
        field = evaluate_field(splauts, torch.linspace(0, 0.6, 61), torch.linspace(0, 4000, 40))
        self.assertEqual(field.shape, (2, 61, 40))
        self.assertTrue(torch.isfinite(field).all() and (field >= 0).all())

    def test_autodiff_matches_finite_differences_for_all_physical_parameters(self):
        dtype = torch.float64
        config = Config()
        stations = station_layout().to(dtype)
        times = torch.linspace(0.2, 1.4, 17, dtype=dtype)
        frequencies = torch.linspace(900, 1800, 9, dtype=dtype)
        values = [torch.tensor(x, dtype=dtype, requires_grad=True) for x in
                  ([[110., 145.]], [0.17], [1.2], [[0.19]], [[1370.]], [[0.08]], [[180.]], [[2.]])]
        def forward(position, onset, amplitude, time, frequency, time_width, frequency_width, weight):
            return render(Splauts(time, frequency, time_width, frequency_width, weight),
                          position, onset, amplitude, stations, times, frequencies, config)
        self.assertTrue(torch.autograd.gradcheck(forward, tuple(values), eps=1e-5, atol=1e-8, rtol=2e-3))
        gradients = torch.autograd.grad(forward(*values).square().sum(), values)
        for gradient in gradients:
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().max()), 1e-18)

    def test_continuous_subframe_delay_and_attenuation(self):
        config = Config()
        splauts = Splauts(*[torch.tensor([x]) for x in ([0.2], [1000.], [0.03], [40.], [2.])])
        position = torch.tensor([[150., 150.]])
        station = torch.tensor([[120., 150.]])
        arrival = 0.1 + 30 / config.sound_speed + 0.2
        args = (splauts, position, torch.tensor([0.1]), torch.ones(1), station)
        peak = render(*args, torch.tensor([arrival]), torch.tensor([1000.]), config)
        self.assertAlmostEqual(float(peak), 2 / 31**2, places=8)
        shifted = render(splauts, position, torch.tensor([0.101]), torch.ones(1), station,
                         torch.tensor([arrival]), torch.tensor([1000.]), config)
        self.assertLess(float(shifted), float(peak))
        # A 1 ms onset shift is below the 10 ms hop and must remain observable.
        self.assertGreater(float(peak - shifted), 1e-8)

    def test_station_event_permutations_and_composition(self):
        torch.manual_seed(12)
        config = Config(splauts=8)
        generator = SplautGenerator(config)
        splauts = generator(torch.randn(2, config.latent_dim), torch.tensor([0.4, 0.4]))
        positions, onsets, amplitudes = torch.tensor([[120., 80.], [190., 170.]]), torch.tensor([0.1, 0.3]), torch.tensor([1., 0.8])
        stations, times, frequencies = station_layout(), torch.linspace(0, 2, 41), torch.linspace(0, 4000, 25)
        prediction = render(splauts, positions, onsets, amplitudes, stations, times, frequencies, config)
        order = torch.tensor([2, 0, 3, 1])
        reordered = render(splauts, positions, onsets, amplitudes, stations[order], times, frequencies, config)
        torch.testing.assert_close(reordered, prediction[order])
        reversed_splauts = Splauts(**{name: value.flip(0) for name, value in vars(splauts).items()})
        reordered = render(reversed_splauts, positions.flip(0), onsets.flip(0), amplitudes.flip(0),
                           stations, times, frequencies, config)
        torch.testing.assert_close(reordered, prediction)
        components = []
        for k in range(2):
            component = Splauts(**{name: value[k:k+1] for name, value in vars(splauts).items()})
            components.append(render(component, positions[k:k+1], onsets[k:k+1], amplitudes[k:k+1],
                                      stations, times, frequencies, config))
        torch.testing.assert_close(prediction, sum(components))


if __name__ == "__main__":
    unittest.main()
