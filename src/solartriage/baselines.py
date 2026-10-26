"""
Non-deep-learning baselines, evaluated on exactly the same splits as the CNN.

* Hot-spot rule  : the classic thermography heuristic "flag a module if its hottest region is
                   much warmer than the module as a whole". Score = 99th percentile - median
                   intensity. Binary (anomaly / normal) only.
* Logistic Reg.  : multinomial logistic regression on standardised raw pixels.
* Random Forest  : 300 trees on raw pixels + simple thermal statistics. Corley et al. (2025)
                   report that a Random Forest reaches 84.0 % accuracy on this dataset, so it is
                   the strongest classical reference point.
Hyper-parameters are chosen on the VALIDATION split by macro-F1.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def hotspot_score(images: np.ndarray) -> np.ndarray:
    flat = images.reshape(len(images), -1).astype(np.float32)
    return np.percentile(flat, 99, axis=1) - np.median(flat, axis=1)


def thermal_features(images: np.ndarray) -> np.ndarray:
    """Pixels + per-image statistics + row/column profiles (captures 1/3-module diode strips)."""
    x = images.astype(np.float32) / 255.0
    flat = x.reshape(len(x), -1)
    stats = np.stack([flat.mean(1), flat.std(1), flat.min(1), flat.max(1),
                      np.percentile(flat, 99, 1) - np.median(flat, 1),
                      np.percentile(flat, 90, 1) - np.percentile(flat, 10, 1)], axis=1)
    return np.hstack([flat, stats, x.mean(2), x.mean(1)])


def fit_logistic(x_tr, y_tr, x_va, y_va, seed: int):
    best = None
    for c in (0.01, 0.1, 1.0):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000, class_weight="balanced",
                                                              random_state=seed))
        m.fit(x_tr.reshape(len(x_tr), -1), y_tr)
        f1 = f1_score(y_va, m.predict(x_va.reshape(len(x_va), -1)), average="macro")
        print(f"  logistic C={c}: val macro-F1 {f1:.4f}", flush=True)
        if best is None or f1 > best[0]:
            best = (f1, c, m)
    return best[2], {"C": best[1], "val_macro_f1": best[0]}


def fit_random_forest(x_tr, y_tr, x_va, y_va, seed: int, n_jobs: int = 4):
    best = None
    f_tr, f_va = thermal_features(x_tr), thermal_features(x_va)
    for leaf in (1, 3):
        m = RandomForestClassifier(n_estimators=300, min_samples_leaf=leaf, class_weight="balanced_subsample",
                                   n_jobs=n_jobs, random_state=seed)
        m.fit(f_tr, y_tr)
        f1 = f1_score(y_va, m.predict(f_va), average="macro")
        print(f"  random forest min_samples_leaf={leaf}: val macro-F1 {f1:.4f}", flush=True)
        if best is None or f1 > best[0]:
            best = (f1, leaf, m)
    return best[2], {"min_samples_leaf": best[1], "val_macro_f1": best[0]}
