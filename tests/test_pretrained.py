import unittest
from unittest.mock import patch

import torch

from splauts.config import Config, station_layout
from splauts.model.generator import evaluate_field
from splauts.simulator.calls import CallSpec
from splauts.training.frozen import fit_frozen_prior
from splauts.training.mixtures import MixtureObservations, mixture_losses, normalized_splauts
from splauts.training.pretrain import FrozenCallPrior, decoder_fingerprint, encode_calls, pretrain_call_prior
from splauts.model.renderer import render


class PretrainedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.set_flush_denormal(True)
        cls.config = Config(splauts=16, latent_dim=8)
        cls.training = pretrain_call_prior(cls.config, per_family=3, steps=200, batch_size=9,
                                           frequency_step=400., log_every=0)

    def test_supervised_learning_and_independent_validation(self):
        training = self.training
        self.assertLess(sum(training.train_nmse) / len(training.train_nmse),
                        training.training_history[0]["shape_nmse"] * 0.5)
        self.assertFalse(set(training.calls).intersection(training.validation_calls))
        self.assertTrue(all(torch.isfinite(torch.tensor(training.validation_nmse))))
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in training.prior.generator.parameters()))

    def test_frequency_translation_and_latent_gradients_survive_freezing(self):
        prior, config = self.training.prior, self.config
        codes = encode_calls([CallSpec("tonal", frequency=1400.), CallSpec("tonal", frequency=1500.)], config.latent_dim)
        durations = torch.full((2,), config.call_duration)
        splauts = prior.generator(codes, durations)
        torch.testing.assert_close(splauts.frequency[1] - splauts.frequency[0], torch.full((config.splauts,), 100.), atol=2e-4, rtol=1e-5)
        torch.testing.assert_close(splauts.time[0], splauts.time[1])
        codes.requires_grad_()
        field = evaluate_field(normalized_splauts(prior.generator, codes, durations, prior.relative_times,
                                                   prior.frequencies), prior.relative_times, prior.frequencies)
        centroid = (field * prior.frequencies[None, None]).sum() / field.sum()
        centroid.backward()
        self.assertTrue(torch.isfinite(codes.grad).all())
        self.assertGreater(float(codes.grad[:, 3].abs().sum()), 1.)
        self.assertTrue(all(p.grad is None for p in prior.generator.parameters()))

    def test_rejects_a_time_grid_incompatible_with_the_prior(self):
        prior = self.training.prior
        times = torch.arange(121) * 0.02
        observed = MixtureObservations(torch.ones(4, len(times), len(prior.frequencies)),
                                       station_layout(), times, prior.frequencies)
        with self.assertRaisesRegex(ValueError, "time step"):
            fit_frozen_prior(observed, prior, self.config, 2, steps=0, proposal_steps=0)

    def test_mixture_inversion_does_not_update_or_retrain_decoder(self):
        config, trained = self.config, self.training.prior
        bank_calls = ([CallSpec("tonal", frequency=f) for f in (1375., 1400., 1425., 1450.)]
                      + [CallSpec("sweep", frequency=f) for f in (2050., 2075., 2100., 2125.)])
        bank = encode_calls(bank_calls, config.latent_dim)
        prior = FrozenCallPrior(trained.generator, bank, trained.relative_times, trained.frequencies, trained.fingerprint)
        unknown = encode_calls([CallSpec("tonal", frequency=1430.), CallSpec("sweep", frequency=2070.)], config.latent_dim)
        durations = torch.full((2,), config.call_duration)
        splauts = normalized_splauts(prior.generator, unknown, durations, prior.relative_times, prior.frequencies).detach()
        stations = station_layout()
        times = torch.arange(241) * 0.01
        positions = torch.tensor([[100., 140.], [210., 180.]])
        power = render(splauts, positions, torch.tensor([0.2, 0.5]), torch.tensor([0.7, 0.4]),
                       stations, times, prior.frequencies, config)
        observed = MixtureObservations(power, stations, times, prior.frequencies)
        before = decoder_fingerprint(prior.generator)
        weights_before = {name: value.clone() for name, value in prior.generator.state_dict().items()}
        with patch("splauts.training.pretrain.synthesize_call", side_effect=AssertionError("No test source access")), \
             patch("splauts.training.pretrain.pretrain_call_prior", side_effect=AssertionError("No retraining")):
            result = fit_frozen_prior(observed, prior, config, 2, steps=100, proposal_steps=60,
                                      proposal_grid_size=13, log_every=0)
        self.assertEqual(before, decoder_fingerprint(prior.generator))
        self.assertEqual(result.decoder_fingerprint, before)
        for name, value in prior.generator.state_dict().items():
            torch.testing.assert_close(value, weights_before[name], rtol=0, atol=0)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in prior.generator.parameters()))
        power_nmse, _ = mixture_losses(result.prediction, power)
        # This checks useful refinement and frozen state, rather than a global
        # convergence guarantee for the deliberately small, sparse test prior.
        # Off-grid localization accuracy is evaluated by the held-out sweep.
        self.assertLess(float(power_nmse), result.power_losses[0] * 0.8)
        nearest = torch.cdist(result.state.latents[:, 3:7], bank[:, 3:7]).min(1).values
        self.assertGreater(float(nearest.max()), 1e-4)


if __name__ == "__main__":
    unittest.main()
