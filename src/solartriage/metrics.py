"""
Evaluation utilities.

Key idea: the dataset is 50 % anomalous, but real solar farms are about 2.2 % anomalous
(Millendorf et al., 2020). Precision and the number of modules an expert must review depend
on that prevalence, so we re-weight the per-class score distributions measured on the test
set to the deployment prevalence instead of reporting accuracy on the balanced test set.

For a threshold t with true-positive rate TPR(t) and false-positive rate FPR(t):
    review workload  W(t)  = pi * TPR + (1 - pi) * FPR      (share of ALL modules flagged)
    precision        P(t)  = pi * TPR / W(t)
    missed anomalies per 10k modules = 10_000 * pi * (1 - TPR)
    false alarms     per 10k modules = 10_000 * (1 - pi) * FPR
Assumption (stated in the report): the score distribution of normal and of anomalous modules is
the same in the field as in this dataset; only the mixing proportion changes.
"""
from __future__ import annotations

import numpy as np
from scipy.special import softmax
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             f1_score, roc_auc_score)

from solartriage import NORMAL_IDX


def anomaly_score(probs: np.ndarray) -> np.ndarray:
    """P(any anomaly) = 1 - P(No-Anomaly)."""
    return 1.0 - probs[:, NORMAL_IDX]


def rates_at(scores: np.ndarray, y_bin: np.ndarray, threshold: float) -> tuple[float, float]:
    flag = scores >= threshold
    tpr = flag[y_bin == 1].mean()
    fpr = flag[y_bin == 0].mean()
    return float(tpr), float(fpr)


def threshold_for_recall(scores: np.ndarray, y_bin: np.ndarray, target_recall: float) -> float:
    """Largest threshold whose anomaly recall is >= target (chosen on VALIDATION data only)."""
    pos = np.sort(scores[y_bin == 1])
    k = int(np.floor((1 - target_recall) * len(pos)))
    return float(pos[k])


def deployment_view(tpr: float, fpr: float, prevalence: float) -> dict:
    workload = prevalence * tpr + (1 - prevalence) * fpr
    return {
        "recall": tpr,
        "fpr": fpr,
        "review_workload": workload,
        "precision_at_prevalence": prevalence * tpr / workload if workload > 0 else float("nan"),
        "missed_per_10k": 10_000 * prevalence * (1 - tpr),
        "false_alarms_per_10k": 10_000 * (1 - prevalence) * fpr,
        "workload_reduction_vs_manual": 1 - workload,
    }


def ap_at_prevalence(scores: np.ndarray, y_bin: np.ndarray, prevalence: float) -> float:
    """Average precision after re-weighting negatives so that positives make up `prevalence`."""
    w = np.where(y_bin == 1, prevalence / y_bin.mean(), (1 - prevalence) / (1 - y_bin.mean()))
    return float(average_precision_score(y_bin, scores, sample_weight=w))


def precision_recall_at_prevalence(scores, y_bin, prevalence, n=400):
    ts = np.unique(np.quantile(scores[y_bin == 1], np.linspace(0, 1, n)))
    out = [deployment_view(*rates_at(scores, y_bin, t), prevalence) for t in ts]
    return np.array([o["recall"] for o in out]), np.array([o["precision_at_prevalence"] for o in out]), \
        np.array([o["review_workload"] for o in out])


def multiclass_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
    }


def binary_summary(scores: np.ndarray, y_bin: np.ndarray, prevalence: float) -> dict:
    return {
        "roc_auc": roc_auc_score(y_bin, scores),
        "ap_dataset_prevalence": average_precision_score(y_bin, scores),
        "ap_real_prevalence": ap_at_prevalence(scores, y_bin, prevalence),
    }


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature scaling (Guo et al., 2017): one scalar T fitted on validation logits by NLL."""
    grid = np.exp(np.linspace(np.log(0.25), np.log(8), 300))
    nll = [-np.log(softmax(logits / t, axis=1)[np.arange(len(y)), y] + 1e-12).mean() for t in grid]
    return float(grid[int(np.argmin(nll))])


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(ece)


# ---------------------------------------------------------------------------
# Uncertainty of test metrics
# ---------------------------------------------------------------------------
def bootstrap_ci(metric_fn, n: int, n_resamples: int, seed: int, stratify: np.ndarray | None = None):
    """95 % percentile bootstrap interval for metric_fn(indices)."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_resamples):
        if stratify is None:
            idx = rng.integers(0, n, n)
        else:  # resample within each class so rare classes are always present
            idx = np.concatenate([rng.choice(np.flatnonzero(stratify == c), (stratify == c).sum())
                                  for c in np.unique(stratify)])
        vals.append(metric_fn(idx))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
