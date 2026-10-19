import unittest
from dataclasses import replace

import torch

from splauts.config import Config, station_layout, validate_stations
from splauts.simulator.calls import CallSpec, power_spectrogram, synthesize_call
from splauts.simulator.scene import SourceEvent, fractional_delay, simulate_scene


class SimulatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_fractional_delay_and_zero_extension(self):
        impulse = torch.tensor([1., 0., 0.])
        actual = fractional_delay(impulse, 0.15, 7, 10)
        self.assertAlmostEqual(float(actual[1]), float(actual[2]), places=6)
        self.assertGreater(float(actual[1]), 0.63)  # Bandlimited half-sample impulse.
        torch.testing.assert_close(fractional_delay(impulse, 0., 5, 10),
                                   torch.tensor([1., 0., 0., 0., 0.]))
        torch.testing.assert_close(fractional_delay(impulse, 0.2, 7, 10),
                                   torch.tensor([0., 0., 1., 0., 0., 0., 0.]), atol=1e-6, rtol=1e-6)
        # Verify fractional delays preserve amplitude, rather than low-pass filtering.
        t = torch.arange(800) / 8000
        wave = torch.sin(2 * torch.pi * 1800 * t)
        delayed = fractional_delay(wave, 0.0100625, 1000, 8000)
        expected = torch.sin(2 * torch.pi * 1800 * (torch.arange(1000) / 8000 - 0.0100625))
        torch.testing.assert_close(delayed[200:700], expected[200:700], atol=0.003, rtol=0.003)

    def test_geometry_design_minimum_and_layouts(self):
        for count in (4, 6, 8):
            for kind in ("corners", "irregular", "clustered"):
                stations = station_layout(count, kind)
                self.assertEqual(stations.shape, (count, 2))
                validate_stations(stations)
        for stations in (torch.zeros(3, 2), torch.arange(8.).reshape(4, 2), torch.zeros(4, 2)):
            with self.assertRaises(ValueError):
                validate_stations(stations)

    def test_delays_gains_truth_and_waveform_noise(self):
        config = Config()
        stations = station_layout()
        call = CallSpec()
        event = SourceEvent(torch.tensor([90., 130.]), 0.2, call, amplitude=1.7)
        scene = simulate_scene([event], stations, config)
        distances = torch.linalg.vector_norm(stations - event.position, dim=-1)
        torch.testing.assert_close(scene.delays[:, 0], distances / 343)
        torch.testing.assert_close(scene.power_gains[:, 0], (distances + 1).pow(-2))
        source = synthesize_call(call)
        for m in range(4):
            expected = fractional_delay(source, event.onset + distances[m] / 343,
                                         scene.waveforms.shape[-1], config.sample_rate)
            torch.testing.assert_close(scene.waveforms[m], expected * 1.7 / (distances[m] + 1))
        torch.testing.assert_close(scene.source_waveforms[0], source)
        torch.testing.assert_close(scene.source_power[0], power_spectrogram(source, config)[0])
        noisy_config = replace(config, noise_std=0.001)
        noisy = simulate_scene([event], stations, noisy_config)
        again = simulate_scene([event], stations, noisy_config)
        torch.testing.assert_close(noisy.waveforms, again.waveforms, rtol=0, atol=0)
        self.assertAlmostEqual(float((noisy.waveforms - scene.waveforms).std()), 0.001, delta=0.00002)

    def test_overlapping_events_mix_before_squaring(self):
        config = Config()
        stations = station_layout()
        event = SourceEvent(torch.tensor([150., 150.]), 0.2, CallSpec())
        single = simulate_scene([event], stations, config)
        pair = simulate_scene([event, event], stations, config)
        torch.testing.assert_close(pair.waveforms, single.waveforms * 2)
        # Coherent overlap exposes why additive power is only an approximation.
        torch.testing.assert_close(pair.power, single.power * 4)
        with self.assertRaises(ValueError):
            simulate_scene([replace(event, onset=2.)], stations, config)


if __name__ == "__main__":
    unittest.main()
