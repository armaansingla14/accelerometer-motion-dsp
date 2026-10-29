import numpy as np
import pytest
from scipy import signal

from motiondsp import filters


def test_moving_average_matches_convolution():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    n = 7
    y = filters.moving_average(x, n)
    ref = np.convolve(x, np.ones(n) / n)[: x.size]
    np.testing.assert_allclose(y[n - 1:], ref[n - 1:], atol=1e-12)
    # start-up: average of the samples seen so far
    np.testing.assert_allclose(y[2], x[:3].mean())


def test_moving_average_2d():
    x = np.arange(20, dtype=float).reshape(10, 2)
    y = filters.moving_average(x, 3)
    np.testing.assert_allclose(y[5], x[3:6].mean(axis=0))


def test_ema_step_response_and_cutoff():
    fs, fc = 100.0, 5.0
    a = filters.ema_alpha(fc, fs)
    y = filters.ema(np.ones(500), a, y0=0.0)
    assert y[-1] == pytest.approx(1.0, abs=1e-9)
    w, h = signal.freqz([a], [1, -(1 - a)], worN=[fc], fs=fs)
    assert 20 * np.log10(abs(h[0])) == pytest.approx(-3.0, abs=0.4)


@pytest.mark.parametrize("order,fc", [(2, 5.0), (4, 12.0), (6, 30.0)])
def test_butterworth_design_matches_scipy(order, fc):
    fs = 100.0
    ours = filters.butter_lowpass_sos(order, fc, fs)
    ref = signal.butter(order, fc, fs=fs, output="sos")
    w = np.linspace(0, fs / 2, 256, endpoint=False)
    _, h1 = signal.sosfreqz(ours, worN=w, fs=fs)
    _, h2 = signal.sosfreqz(ref, worN=w, fs=fs)
    np.testing.assert_allclose(np.abs(h1), np.abs(h2), atol=1e-9)
    _, hc = signal.sosfreqz(ours, worN=[fc], fs=fs)
    assert 20 * np.log10(abs(hc[0])) == pytest.approx(-3.01, abs=0.02)


def test_butterworth_rejects_bad_args():
    with pytest.raises(ValueError):
        filters.butter_lowpass_sos(3, 10, 100)
    with pytest.raises(ValueError):
        filters.butter_lowpass_sos(4, 60, 100)


def test_df2t_reference_matches_scipy():
    rng = np.random.default_rng(1)
    x = rng.normal(size=300)
    sos = filters.butter_lowpass_sos(4, 10.0, 100.0)
    np.testing.assert_allclose(filters.sosfilt_df2t(sos, x), signal.sosfilt(sos, x), atol=1e-12)


def test_steady_state_init_has_no_transient():
    sos = filters.butter_lowpass_sos(4, 10.0, 100.0)
    x = np.full(100, 9.81)
    np.testing.assert_allclose(filters.sosfilt_steady(sos, x), x, atol=1e-9)
    x2 = np.column_stack([x, -x])
    np.testing.assert_allclose(filters.sosfilt_steady(sos, x2), x2, atol=1e-9)


def test_hampel_removes_spikes_but_keeps_signal():
    t = np.arange(400) / 100
    clean = np.sin(2 * np.pi * 1.0 * t)
    x = clean + np.random.default_rng(0).normal(0, 0.05, t.size)
    x[[50, 120, 300]] += [8, -6, 7]
    y = filters.hampel(x, 7, 3.0)
    assert np.max(np.abs(y - clean)) < 0.3
    # the vast majority of non-spike samples are passed through unchanged
    assert np.mean(y == x) > 0.75


def test_median_filter_odd_only():
    with pytest.raises(ValueError):
        filters.median_filter(np.zeros(10), 4)
    np.testing.assert_allclose(filters.median3(np.array([0, 0, 9, 0, 0.0])), 0)
