from typing import Any
import warnings
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.exceptions import ConvergenceWarning

# Suppress warnings for cleaner progress bars
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def _safe_evaluate_model(X_train, y_train, X_test, y_true, **kwargs):
    """Safely evaluate a model configuration"""
    try:
        classifier = KNeighborsClassifier(**kwargs)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        return accuracy, f1, True
    except Exception:
        return 0.0, 0.0, False


def _optimize_nn(X_train, y_train, X_test, y_true, opts):
    """Optimize number of neighbors"""
    max_acc = -np.inf
    best_f1 = 0.0
    # More intelligent range - most optimal k values are usually small
    variable_array = (
        list(range(1, 21)) + list(range(25, 101, 5)) + list(range(110, 201, 10))
    )
    best_val = variable_array[0]

    with tqdm(
        variable_array, desc="Optimizing Number of Neighbors", leave=False
    ) as pbar:
        for v in pbar:
            test_opts = opts.copy()
            test_opts["n_neighbors"] = v

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, **test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_val = v

            pbar.set_postfix(
                {
                    "k": v,
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts["n_neighbors"] = best_val
    return opts, max_acc, best_f1


def _optimize_wt(X_train, y_train, X_test, y_true, opts):
    """Optimize weights"""
    max_acc = -np.inf
    best_f1 = 0.0
    variable_array = ["uniform", "distance"]
    best_val = variable_array[0]

    with tqdm(variable_array, desc="Optimizing Weights", leave=False) as pbar:
        for v in pbar:
            test_opts = opts.copy()
            test_opts["weights"] = v

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, **test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_val = v

            pbar.set_postfix(
                {
                    "weights": v,
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts["weights"] = best_val
    return opts, max_acc, best_f1


def _optimize_algo(X_train, y_train, X_test, y_true, opts):
    """Optimize algorithm"""
    max_acc = -np.inf
    best_f1 = 0.0
    variable_array = ["auto", "ball_tree", "kd_tree", "brute"]
    best_val = variable_array[0]

    with tqdm(variable_array, desc="Optimizing Algorithm", leave=False) as pbar:
        for v in pbar:
            test_opts = opts.copy()
            test_opts["algorithm"] = v

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, **test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_val = v

            pbar.set_postfix(
                {
                    "algo": v[:8],
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts["algorithm"] = best_val
    return opts, max_acc, best_f1


def _optimize_ls(X_train, y_train, X_test, y_true, opts):
    """Optimize leaf size"""
    max_acc = -np.inf
    best_f1 = 0.0
    # More efficient range for leaf size
    variable_array = [1, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100]
    best_val = variable_array[0]

    with tqdm(variable_array, desc="Optimizing Leaf Size", leave=False) as pbar:
        for v in pbar:
            test_opts = opts.copy()
            test_opts["leaf_size"] = v

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, **test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_val = v

            pbar.set_postfix(
                {
                    "leaf_size": v,
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts["leaf_size"] = best_val
    return opts, max_acc, best_f1


def _optimize_p(X_train, y_train, X_test, y_true, opts):
    """Optimize p parameter (power parameter for Minkowski metric)"""
    max_acc = -np.inf
    best_f1 = 0.0
    # Focus on more commonly used p values
    variable_array = [1, 1.2, 1.5, 2, 2.5, 3, 4, 5]  # More efficient range
    best_val = variable_array[0]

    with tqdm(variable_array, desc="Optimizing P Parameter", leave=False) as pbar:
        for v in pbar:
            # Only optimize p if metric is minkowski or compatible
            if opts["metric"] not in ["minkowski"]:
                pbar.set_postfix(
                    {"p": f"{v:.1f}", "status": "skipped (incompatible metric)"}
                )
                continue

            test_opts = opts.copy()
            test_opts["p"] = v

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, **test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_val = v

            pbar.set_postfix(
                {
                    "p": f"{v:.1f}",
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts["p"] = best_val
    return opts, max_acc, best_f1


def _optimize_metric(X_train, y_train, X_test, y_true, opts):
    """Optimize distance metric"""
    max_acc = -np.inf
    best_f1 = 0.0

    # More curated list of commonly effective metrics
    variable_array = [
        "euclidean",
        "manhattan",
        "minkowski",
        "chebyshev",
        "cosine",
        "correlation",
        "hamming",
        "jaccard",
        "braycurtis",
        "canberra",
        "cityblock",
        "dice",
        "rogerstanimoto",
        "russellrao",
        "sokalmichener",
        "sokalsneath",
        "yule",
    ]
    best_val = variable_array[0]

    with tqdm(variable_array, desc="Optimizing Distance Metric", leave=False) as pbar:
        for v in pbar:
            test_opts = opts.copy()
            test_opts["metric"] = v

            # Reset p to 2 for non-minkowski metrics to avoid conflicts
            if v != "minkowski":
                test_opts["p"] = 2

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, **test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_val = v

            pbar.set_postfix(
                {
                    "metric": v[:10],
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts["metric"] = best_val
    return opts, max_acc, best_f1


def _optimize_knn(X_train, y_train, X_test, y_true, cycles=2):
    """
    Optimizes the hyperparameters for KNeighborsClassifier.
    :param X_train: Training data
    :param y_train: Training labels
    :param X_test: Test data
    :param y_true: True labels for the test data
    :param cycles: Number of optimization cycles
    """

    # Define initial parameters
    opts = {
        "n_neighbors": 5,
        "weights": "uniform",
        "algorithm": "auto",
        "leaf_size": 30,
        "p": 2,
        "metric": "minkowski",
        "metric_params": None,
        "n_jobs": -1,  # Use all available cores
    }

    # Optimize hyperparameters
    ma_vec = []
    f1_vec = []

    # Main optimization loop with overall progress bar
    with tqdm(range(cycles), desc="Optimization Cycles", position=0) as cycle_pbar:
        for c in cycle_pbar:
            cycle_pbar.set_description(f"KNN Cycle {c + 1}/{cycles}")

            opts, _, _ = _optimize_nn(X_train, y_train, X_test, y_true, opts)
            opts, _, _ = _optimize_wt(X_train, y_train, X_test, y_true, opts)
            opts, _, _ = _optimize_algo(X_train, y_train, X_test, y_true, opts)
            opts, _, _ = _optimize_ls(X_train, y_train, X_test, y_true, opts)
            opts, _, _ = _optimize_p(X_train, y_train, X_test, y_true, opts)
            opts, ma, f1 = _optimize_metric(X_train, y_train, X_test, y_true, opts)

            ma_vec.append(ma)
            f1_vec.append(f1)

            cycle_pbar.set_postfix(
                {
                    "accuracy": f"{ma:.4f}",
                    "f1": f"{f1:.4f}",
                    "best_overall": f"{max(ma_vec):.4f}",
                    "k": opts["n_neighbors"],
                    "metric": opts["metric"][:8],
                }
            )

    return opts, ma, f1, ma_vec, f1_vec
