"""Checks that the C++ WindowProcessor reproduces the Python features.

Skipped when cpp/build/extract_features has not been built.
"""

import numpy as np
import pytest

from motiondsp import cpp_bridge, pipeline as P, synth

pytestmark = pytest.mark.skipif(cpp_bridge.tool("extract_features") is None,
                                reason="C++ tools not built")


def test_cpp_features_match_python(tmp_path):
    cfg = P.PipelineConfig()
    sessions = synth.generate_dataset(2, seed=11)
    X, _, _, _, wins, biases = P.build_feature_table(sessions, cfg)
    wins, biases, X = wins[:40], biases[:40], X[:40]
    txt = tmp_path / "w.txt"
    cpp_bridge.write_windows(txt, wins, biases)
    Xc = cpp_bridge.run_extract_features(txt, tmp_path / "f.csv", cfg.lp_cutoffs)
    assert Xc.shape == X.shape
    np.testing.assert_allclose(Xc, X, rtol=1e-6, atol=1e-6)
