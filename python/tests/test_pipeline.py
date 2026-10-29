import numpy as np
import pytest

from motiondsp import classify, pipeline as P, synth


@pytest.fixture(scope="module")
def sessions():
    return synth.generate_dataset(6, seed=7)


def test_session_shapes_and_labels(sessions):
    s = sessions[0]
    assert s.clean.shape == s.noisy.shape == (s.n, 6)
    assert s.labels.min() >= 0 and s.labels.max() < len(synth.CLASSES)
    # leading calibration period is stationary
    assert np.all(s.labels[: s.calib_samples] == 0)
    np.testing.assert_allclose(s.clean[: s.calib_samples, 2], synth.G, atol=1e-9)


def test_generator_is_deterministic():
    a = synth.generate_dataset(2, seed=3)
    b = synth.generate_dataset(2, seed=3)
    np.testing.assert_array_equal(a[1].noisy, b[1].noisy)


def test_bias_estimate_is_close(sessions):
    s = sessions[0]
    bias = P.estimate_bias(s.noisy, s.calib_samples)
    true_bias = (s.noisy - s.clean)[: s.calib_samples, :4].mean(axis=0)
    np.testing.assert_allclose(bias[:4], true_bias, atol=1e-2)
    assert np.all(bias[4:] == 0)


@pytest.mark.parametrize("method", ["ema", "hampel+ema", "hampel+butter", "zerophase"])
def test_filters_reduce_error_on_gyro(sessions, method):
    cfg = P.PipelineConfig()
    num = den = 0.0
    for s in sessions:
        b = P.estimate_bias(s.noisy, s.calib_samples)
        est = P.denoise_stream(s.noisy, b, cfg, method, param=8.0)
        num += P.rmse_per_channel(est, s.clean)[3] ** 2
        den += P.rmse_per_channel(s.noisy, s.clean)[3] ** 2
    assert num < den


def test_unknown_method_raises(sessions):
    s = sessions[0]
    with pytest.raises(ValueError):
        P.denoise_stream(s.noisy, np.zeros(6), P.PipelineConfig(), "kalman")


def test_window_label_purity():
    assert P.window_label(np.array([1] * 10), 0.9) == 1
    assert P.window_label(np.array([1] * 5 + [2] * 5), 0.9) == -1


def test_feature_table_and_classifier(sessions):
    cfg = P.PipelineConfig()
    X, Xb, y, g, wins, biases = P.build_feature_table(sessions, cfg)
    assert X.shape[1] == len(P.FEATURE_NAMES)
    assert Xb.shape[1] == len(P.BASELINE_NAMES)
    assert np.isfinite(X).all()
    tr, te = classify.session_split(g, 0.34, seed=0)
    assert set(g[tr]).isdisjoint(set(g[te]))
    r = classify.evaluate(classify.make_model("rf"), X[tr], y[tr], X[te], y[te], 5)
    assert r.accuracy > 0.5
    assert r.confusion.sum() == te.sum()
