from typing import Any
import warnings
import numpy as np
from tqdm.auto import tqdm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score
from sklearn.exceptions import ConvergenceWarning

# Suppress warnings for cleaner progress bars
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def _safe_evaluate_model(X_train, y_train, X_test, y_true, **kwargs):
    """Safely evaluate a model configuration"""
    try:
        classifier = LinearDiscriminantAnalysis(**kwargs)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        return accuracy, f1, True
    except Exception as e:
        return 0.0, 0.0, False


def _optimize_sl(X_train, y_train, X_test, y_true, opts):
    """Optimize solver"""
    max_acc = -np.inf
    best_f1 = 0.0
    variable_array = ["svd", "lsqr", "eigen"]
    best_val = variable_array[0]

    with tqdm(variable_array, desc="Optimizing Solver", leave=False) as pbar:
        for v in pbar:
            test_opts = opts.copy()
            test_opts["solver"] = v

            # Handle solver-specific constraints
            if v == "svd":
                test_opts["shrinkage"] = None  # SVD doesn't support shrinkage
            elif v == "eigen" and test_opts["shrinkage"] == "auto":
                test_opts["shrinkage"] = (
                    None  # Eigen with auto shrinkage can be problematic
                )

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, **test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_val = v

            pbar.set_postfix(
                {
                    "solver": v,
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts["solver"] = best_val
    return opts, max_acc, best_f1


def _optimize_sh(X_train, y_train, X_test, y_true, opts):
    """Optimize shrinkage"""
    max_acc = -np.inf
    best_f1 = 0.0

    # Skip shrinkage optimization if solver is SVD (doesn't support shrinkage)
    if opts["solver"] == "svd":
        return opts, max_acc, best_f1

    # More focused shrinkage values
    variable_array = [None, "auto"] + list(np.arange(0.0, 1.01, 0.1))
    best_val = variable_array[0]

    with tqdm(variable_array, desc="Optimizing Shrinkage", leave=False) as pbar:
        for v in pbar:
            test_opts = opts.copy()
            test_opts["shrinkage"] = v

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, **test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_val = v

            pbar.set_postfix(
                {
                    "shrinkage": str(v) if not isinstance(v, float) else f"{v:.1f}",
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts["shrinkage"] = best_val
    return opts, max_acc, best_f1


def _optimize_nc(X_train, y_train, X_test, y_true, opts):
    """Optimize number of components"""
    max_acc = -np.inf
    best_f1 = 0.0

    # Determine reasonable range based on data
    n_classes = len(np.unique(y_train))
    n_features = X_train.shape[1]
    max_components = min(n_classes - 1, n_features)  # LDA constraint

    # Create a reasonable range
    if max_components <= 1:
        variable_array = [None, 1]
    else:
        variable_array = [None] + list(range(1, max_components + 1))

    best_val = variable_array[0]

    with tqdm(
        variable_array, desc="Optimizing Number of Components", leave=False
    ) as pbar:
        for v in pbar:
            test_opts = opts.copy()
            test_opts["n_components"] = v

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, **test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_val = v

            pbar.set_postfix(
                {
                    "n_comp": str(v) if v is not None else "None",
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts["n_components"] = best_val
    return opts, max_acc, best_f1


def _optimize_tol(X_train, y_train, X_test, y_true, opts):
    """Optimize tolerance"""
    max_acc = -np.inf
    best_f1 = 0.0
    variable_array = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    best_val = variable_array[0]

    with tqdm(variable_array, desc="Optimizing Tolerance", leave=False) as pbar:
        for v in pbar:
            test_opts = opts.copy()
            test_opts["tol"] = v

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, **test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_val = v

            pbar.set_postfix(
                {
                    "tol": f"{v:.0e}",
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts["tol"] = best_val
    return opts, max_acc, best_f1


def _optimize_store_cov(X_train, y_train, X_test, y_true, opts):
    """Optimize store_covariance"""
    max_acc = -np.inf
    best_f1 = 0.0
    variable_array = [False, True]
    best_val = variable_array[0]

    with tqdm(variable_array, desc="Optimizing Store Covariance", leave=False) as pbar:
        for v in pbar:
            test_opts = opts.copy()
            test_opts["store_covariance"] = v

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, **test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_val = v

            pbar.set_postfix(
                {
                    "store_cov": str(v),
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts["store_covariance"] = best_val
    return opts, max_acc, best_f1


def _optimize_linear_discriminant(X_train, y_train, X_test, y_true, cycles=2):
    """
    Optimizes the hyperparameters for LinearDiscriminantAnalysis.
    :param X_train: Training data
    :param y_train: Training labels
    :param X_test: Test data
    :param y_true: True labels for the test data
    :param cycles: Number of optimization cycles
    """

    # Define initial parameters
    opts = {
        "solver": "svd",
        "shrinkage": None,
        "priors": None,
        "n_components": None,
        "store_covariance": False,
        "tol": 1e-4,
        "covariance_estimator": None,
    }

    # Optimize hyperparameters
    ma_vec = []
    f1_vec = []

    # Main optimization loop with overall progress bar
    with tqdm(range(cycles), desc="Optimization Cycles", position=0) as cycle_pbar:
        for c in cycle_pbar:
            cycle_pbar.set_description(f"LDA Cycle {c + 1}/{cycles}")

            opts, _, _ = _optimize_sl(X_train, y_train, X_test, y_true, opts)
            opts, _, _ = _optimize_sh(X_train, y_train, X_test, y_true, opts)
            opts, _, _ = _optimize_nc(X_train, y_train, X_test, y_true, opts)
            opts, _, _ = _optimize_tol(X_train, y_train, X_test, y_true, opts)
            opts, ma, f1 = _optimize_store_cov(X_train, y_train, X_test, y_true, opts)

            ma_vec.append(ma)
            f1_vec.append(f1)

            cycle_pbar.set_postfix(
                {
                    "accuracy": f"{ma:.4f}",
                    "f1": f"{f1:.4f}",
                    "best_overall": f"{max(ma_vec):.4f}",
                    "solver": opts["solver"],
                    "shrinkage": (
                        str(opts["shrinkage"])[:6]
                        if opts["shrinkage"] is not None
                        else "None"
                    ),
                }
            )

    return opts, ma, f1, ma_vec, f1_vec
