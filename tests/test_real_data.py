"""Field-data risks: CSV ordering, rational resampling, seams and audio masks."""

import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

try:
    import soundfile as sf
    from scipy import signal
    from splauts.real_data import (read_stations, load_audio, stft_power,
                                   audio_from_mask, window_starts)
    HAS_REAL = True
except ImportError:
    HAS_REAL = False


@unittest.skipUnless(HAS_REAL, "Install neural-splauts[real]")
class RealDataTests(unittest.TestCase):
    def make_metadata(self, path):
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["TrackIndex", "StationID", "Lat", "Lon"])
            # Row order deliberately differs from channel order. Duplicate names
            # away from the anchor are valid; explicit tracks are authoritative.
            for track in [3, 1, 6, 2, 5, 4]:
                writer.writerow([track, "M04" if track == 1 else "duplicate",
                                 .45 + (track//3)*.001, 36.84+(track%3)*.001])

    def test_track_mapping_and_local_metric_coordinates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"stations.csv"
            self.make_metadata(path)
            table = read_stations(path, count=6)
            self.assertEqual(table.tracks[0], 0)
            np.testing.assert_array_equal(np.sort(table.tracks), np.arange(6))
            np.testing.assert_allclose(table.xy[0], 0)
            self.assertTrue(110 < np.linalg.norm(table.xy[1]) < 112)

    def test_streaming_resampling_preserves_channels_delays_and_block_seams(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_metadata(root/"stations.csv")
            table = read_stations(root/"stations.csv", count=6)
            rate = 16000
            t = np.arange(rate*9)/rate
            x = np.stack([(k+1)*.01*np.sin(2*np.pi*29*(t-k*.017))
                          + .01*np.sin(2*np.pi*3000*t) for k in range(6)], axis=1)
            sf.write(root/"audio.wav", x, rate, subtype="DOUBLE")
            actual, info = load_audio(root/"audio.wav", table)
            expected = signal.resample_poly(x[:, table.tracks], 4, 125, axis=0).T
            self.assertEqual(actual.shape, (6, 9*512))
            np.testing.assert_allclose(actual, expected, atol=2e-7)
            self.assertEqual(info["sample_rate"], 16000)
            # Rejecting a 3 kHz alias should leave just the 29 Hz signal.
            theory = (table.tracks[:,None]+1)*.01*np.sin(
                2*np.pi*29*(np.arange(9*512)[None]/512-table.tracks[:,None]*.017))
            np.testing.assert_allclose(actual[:,100:-100], theory[:,100:-100], atol=1e-4)

    def test_stft_masks_conserve_reference_waveform(self):
        x = np.random.default_rng(9).normal(size=(1, 8*512)).astype(np.float32)
        z, times, frequencies = stft_power(x)
        self.assertEqual(z.shape, (1, len(times), len(frequencies)))
        self.assertEqual(times[1]-times[0], 1/16)
        band = (frequencies >= 12) & (frequencies <= 120)
        full = np.broadcast_to(band, z[0].shape)
        mixture = audio_from_mask(z[0], full)
        a = audio_from_mask(z[0], full*.3)
        b = audio_from_mask(z[0], full*.7)
        np.testing.assert_allclose(a+b, mixture, atol=1e-6)
        np.testing.assert_allclose(audio_from_mask(z[0], np.ones_like(z[0])), x[0], atol=1e-6)

    def test_windows_cover_tail_without_out_of_bounds_audio(self):
        np.testing.assert_array_equal(window_starts(19, 8, 4), [0,4,8,11])
        for args in [(7,8,4), (20,8,9), (20,8,0)]:
            with self.assertRaises(ValueError):
                window_starts(*args)


if __name__ == "__main__":
    unittest.main()
