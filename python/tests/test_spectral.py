import numpy as np
import pytest

from motiondsp import spectral


@pytest.mark.parametrize("n", [1, 2, 8, 64, 128, 1024])
def test_fft_matches_numpy(n):
    rng = np.random.default_rng(n)
    x = rng.normal(size=n) + 1j * rng.normal(size=n)
    np.testing.assert_allclose(spectral.fft_radix2(x), np.fft.fft(x), atol=1e-9)


def test_dft_matches_fft():
    rng = np.random.default_rng(3)
    x = rng.normal(size=48)
    np.testing.assert_allclose(spectral.dft(x), np.fft.fft(x), atol=1e-9)


def test_fft_rejects_non_power_of_two():
    with pytest.raises(ValueError):
        spectral.fft_radix2(np.zeros(12))


def test_power_spectrum_parseval_and_peak():
    fs, n = 100.0, 128
    t = np.arange(n) / fs
    f0 = 25.0  # exactly on a bin (25 * 128 / 100 = 32)
    x = 2.0 * np.sin(2 * np.pi * f0 * t)
    f, p = spectral.power_spectrum(x, fs)
    assert spectral.dominant_frequency(f, p) == pytest.approx(f0)
    # Hann-windowed power normalised by sum(w^2) recovers the mean-square value
    assert p.sum() / n == pytest.approx(np.mean(x ** 2), rel=0.02)


def test_band_energies_partition_spectrum():
    rng = np.random.default_rng(0)
    f, p = spectral.power_spectrum(rng.normal(size=128), 100.0)
    bands = spectral.band_energies(f, p)
    assert bands.sum() == pytest.approx(p.sum())
    assert 0 < spectral.spectral_centroid(f, p) < 50
