"""Metrics appropriate for an ordinal age-classification problem.

Plain accuracy treats "predicted 2.5, truth 5.5" and "predicted 2.5, truth
3.5" as equally wrong. For deer aging they are not remotely equally wrong, so
the leaderboard reports ordinal-aware measures alongside accuracy.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score


def ordinal_metrics(y_true, y_pred, class_ages):
    """Score predictions against truth for an ordinal target.

    Args:
        y_true: Integer class indices of the true ages.
        y_pred: Integer class indices of the predicted ages.
        class_ages: Age in years for each class index, ascending, e.g.
            ``[1.5, 2.5, 3.5, 4.5, 5.5]``.

    Returns:
        dict with accuracy, within-one-class accuracy, mean absolute error in
        years, quadratic weighted kappa, and macro F1.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    ages = np.asarray(class_ages, dtype=float)

    if y_true.size == 0:
        raise ValueError("cannot score an empty prediction set")

    exact = y_true == y_pred
    within_one = np.abs(y_true - y_pred) <= 1
    mae_years = float(np.mean(np.abs(ages[y_true] - ages[y_pred])))

    # QWK is undefined when only one class is present in both vectors; fall
    # back to accuracy so the leaderboard still sorts sensibly.
    if len(np.unique(np.concatenate([y_true, y_pred]))) < 2:
        qwk = float(exact.mean())
    else:
        qwk = float(
            cohen_kappa_score(
                y_true, y_pred, weights="quadratic", labels=np.arange(len(ages))
            )
        )

    return {
        "accuracy": float(exact.mean()),
        "within_one": float(within_one.mean()),
        "mae_years": mae_years,
        "qwk": qwk,
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                labels=np.arange(len(ages)),
                zero_division=0,
            )
        ),
        "n": int(y_true.size),
    }


def bootstrap_ci(y_true, y_pred, class_ages, metric="accuracy", n_boot=2000, seed=0):
    """Percentile bootstrap confidence interval for one metric.

    With a test set of only a few dozen images the point estimate is noisy
    enough that reporting it bare is misleading. This quantifies that noise.

    Returns:
        (point_estimate, lo, hi) at the 95% level.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    rng = np.random.default_rng(seed)
    n = y_true.size

    point = ordinal_metrics(y_true, y_pred, class_ages)[metric]

    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        # A resample can be single-class; ordinal_metrics handles that.
        draws.append(ordinal_metrics(y_true[idx], y_pred[idx], class_ages)[metric])

    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def confusion(y_true, y_pred, class_ages):
    """Confusion matrix with rows=truth, cols=prediction, over all classes."""
    return confusion_matrix(
        y_true, y_pred, labels=np.arange(len(class_ages))
    ).tolist()