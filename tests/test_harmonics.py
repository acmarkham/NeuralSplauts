from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import unittest

import torch

from splauts.config import Config, station_layout
from splauts.model.call_prior import CallPriorGenerator, acoustic_values
from splauts.model.generator import Splauts, gaussian_kernel
from splauts.model.renderer import render
from splauts.simulator.calls import CallSpec, synthesize_call
from splauts.simulator.scene import random_scene
from splauts.training.pretrain import (decoder_fingerprint, encode_calls, load_frozen_prior,
                                      sample_training_calls, template_codes)


class HarmonicTests(unittest.TestCase):
    def test_gaussian_tail_cutoff_preserves_values_and_gradients(self):
        x = torch.linspace(-20., 20., 1001, dtype=torch.float64, requires_grad=True)
        expected = torch.exp(-0.5 * x.square())
        actual = gaussian_kernel(x)
        self.assertLess(float((actual - expected).abs().max().detach()), 2e-35)
        gradient, = torch.autograd.grad(actual.sum(), x)
        self.assertLess(float((gradient + x * expected).abs().max().detach()), 3e-34)

    def test_harmonic_spectrum_rolloff_and_equal_energy(self):
        single = CallSpec(duration=1., frequency=400., bandwidth=0.)
        rich = replace(single, harmonic_count=4)
        a, b = synthesize_call(single), synthesize_call(rich)
        self.assertAlmostEqual(float(a.square().sum()), float(b.square().sum()), delta=0.01)
        spectrum = torch.fft.rfft(b).abs()
        peaks = spectrum[torch.tensor([400, 800, 1200, 1600])]
        expected = torch.arange(1., 5.).pow(-rich.harmonic_rolloff)
        torch.testing.assert_close(peaks / peaks[0], expected, atol=1e-5, rtol=1e-4)
        self.assertLess(float(spectrum[2000]), float(peaks[0]) * 1e-5)

    def test_rejects_aliasing_and_invalid_structure(self):
        with self.assertRaisesRegex(ValueError, "Nyquist"):
            synthesize_call(CallSpec(frequency=1600., bandwidth=100., harmonic_count=3))
        with self.assertRaises(ValueError):
            synthesize_call(CallSpec(harmonic_count=0))
        with self.assertRaisesRegex(ValueError, "frequency_scale"):
            CallPriorGenerator(Config(harmonic_count=4))

    def test_paired_scenes_preserve_geometry_and_fundamentals(self):
        config = Config(frequency_scale=0.3)
        single = random_scene(config, 5, 4)
        rich = random_scene(replace(config, harmonic_count=4), 5, 4)
        for a, b in zip(single.events, rich.events):
            torch.testing.assert_close(a.position, b.position)
            self.assertEqual(a.onset, b.onset)
            self.assertEqual(a.amplitude, b.amplitude)
            self.assertEqual(replace(a.call, harmonic_count=4), b.call)

    def test_one_code_controls_and_propagates_every_harmonic(self):
        config = Config(harmonic_count=4, frequency_scale=0.3, splauts=32)
        generator = CallPriorGenerator(config).eval().requires_grad_(False)
        calls = [CallSpec(frequency=f, bandwidth=60., harmonic_count=4) for f in (400., 420.)]
        codes = encode_calls(calls, config.latent_dim, config).requires_grad_()
        torch.testing.assert_close(acoustic_values(codes, config)[:, 0], torch.tensor([400., 420.]))
        decoded = generator(codes, torch.full((2,), 0.4))
        orders = torch.arange(config.splauts) % 4 + 1
        torch.testing.assert_close(decoded.frequency[1] - decoded.frequency[0], orders * 20., atol=2e-4, rtol=1e-5)
        decoded.frequency.sum().backward()
        self.assertGreater(float(codes.grad[:, 3].abs().sum()), 1.)
        self.assertTrue(all(p.grad is None for p in generator.parameters()))
        # A single event axis: shifting its onset shifts all bands together.
        atom = Splauts(**{name: value[:1].detach() for name, value in vars(decoded).items()})
        times, frequencies = torch.arange(241) * 0.01, torch.arange(257) * 15.625
        args = (torch.tensor([1.]), station_layout(), times, frequencies, config)
        a = render(atom, torch.tensor([[130., 140.]]), torch.tensor([0.2]), *args)
        b = render(atom, torch.tensor([[130., 140.]]), torch.tensor([0.22]), *args)
        torch.testing.assert_close(a[:, :-2], b[:, 2:], atol=1e-8, rtol=1e-4)
        for h in range(1, 5):
            self.assertGreater(float(a[..., (frequencies - 400 * h).abs() < 50].sum()), 0.)

    def test_training_domain_and_checkpoint_harmonic_compatibility(self):
        config = Config(harmonic_count=4, frequency_scale=0.3, splauts=16)
        calls = sample_training_calls(config, 2, 23)
        encode_calls(calls, config.latent_dim, config)
        for call in calls:
            self.assertTrue(torch.isfinite(synthesize_call(call)).all())
        generator = CallPriorGenerator(config).eval().requires_grad_(False)
        bundle = dict(config=asdict(config), generator=generator.state_dict(),
                      fingerprint=decoder_fingerprint(generator), bank_codes=template_codes(config, 100.),
                      relative_times=torch.arange(41) * 0.01, frequencies=torch.arange(257) * 15.625)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prior.pt"
            torch.save(bundle, path)
            self.assertEqual(load_frozen_prior(path, config).fingerprint, bundle["fingerprint"])
            for other in (replace(config, harmonic_count=1), replace(config, harmonic_rolloff=1.),
                          replace(config, frequency_scale=0.25)):
                with self.assertRaisesRegex(ValueError, "configuration"):
                    load_frozen_prior(path, other)
