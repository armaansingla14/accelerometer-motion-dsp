"""Motion classifiers and evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_model(kind: str, seed: int = 0):
    if kind == "rf":
        return RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                      n_jobs=2, random_state=seed)
    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    raise ValueError(kind)


def session_split(groups: np.ndarray, test_frac: float, seed: int = 0):
    """Split by session so that no recording contributes to both sets."""
    rng = np.random.default_rng(seed)
    ids = np.unique(groups)
    rng.shuffle(ids)
    n_test = max(1, int(round(test_frac * ids.size)))
    test_ids = set(ids[:n_test].tolist())
    test = np.array([g in test_ids for g in groups])
    return ~test, test


@dataclass
class EvalResult:
    accuracy: float
    macro_f1: float
    confusion: np.ndarray


def evaluate(model, Xtr, ytr, Xte, yte, n_classes: int) -> EvalResult:
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    return EvalResult(
        accuracy=float(accuracy_score(yte, pred)),
        macro_f1=float(f1_score(yte, pred, average="macro")),
        confusion=confusion_matrix(yte, pred, labels=list(range(n_classes))),
    )
