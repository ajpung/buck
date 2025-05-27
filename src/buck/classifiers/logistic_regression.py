from typing import Any
import warnings
import numpy as np
from tqdm.auto import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.exceptions import ConvergenceWarning

# Suppress warnings for cleaner progress bars
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def _safe_evaluate_model(X_train, y_train, X_test, y_true, **kwargs):
    """Safely evaluate a model configuration"""
    try:
        classifier = LogisticRegression(**kwargs)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        return accuracy, f1, True
    except Exception:
        return 0.0, 0.0, False


def _optimize_rs(X_train, y_train, X_test, y_true, opts):
    """Optimize random state"""
    max_acc = -np.inf
    best_f1 = 0.0
    variable_array = np.arange(50)  # Reduced range for efficiency
    best_val = variable_array[0]

    with tqdm(variable_array, desc="Optimizing Random State", leave=False) as pbar:
        for v in pbar:
            test_opts = opts.copy()
            test_opts["random_state"] = v

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, **test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_val = v

            pbar.set_postfix(
                {
                    "rs": v,
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts["random_state"] = best_val
    return opts, max_acc, best_f1


def _optimize_penalty(X_train, y_train, X_test, y_true, opts):
    """Optimize penalty type"""
    max_acc = -np.inf
    best_f1 = 0.0
    variable_array = ["l1", "l2", "elasticnet", None]
    best_val = variable_array[0]

    with tqdm(variable_array, desc="Optimizing Penalty", leave=False) as pbar:
        for v in pbar:
            test_opts = opts.copy()
            test_opts["penalty"] = v

            # Handle solver-penalty compatibility
            if v == "l1":
                test_opts["solver"] = "liblinear"  # L1 works best with liblinear
            elif v == "elasticnet":
                test_opts["solver"] = "saga"  # Elasticnet only works with saga
                if test_opts["l1_ratio"] is None:
                    test_opts["l1_ratio"] = 0.5  # Required for elasticnet
            elif v is None:
                test_opts["solver"] = "lbfgs"  # No penalty works with lbfgs

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, **test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_val = v

            pbar.set_postfix(
                {
                    "penalty": str(v) if v is not None else "None",
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts["penalty"] = best_val
    return opts, max_acc, best_f1


def _optimize_tol(X_train, y_train, X_test, y_true, opts):
    """Optimize tolerance"""
    max_acc = -np.inf
    best_f1 = 0.0
    variable_array = np.logspace(-6, -2, 10)  # More efficient range: 1e-6 to 1e-2
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


def _optimize_c(X_train, y_train, X_test, y_true, opts):
    """Optimize regularization strength (C)"""
    max_acc = -np.inf
    best_f1 = 0.0
    variable_array = np.logspace(-4, 2, 15)  # More efficient range: 1e-4 to 100
    best_val = variable_array[0]

    with tqdm(variable_array, desc="Optimizing C Parameter", leave=False) as pbar:
        for v in pbar:
            test_opts = opts.copy()
            test_opts["C"] = v

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, **test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_val = v

            pbar.set_postfix(
                {
                    "C": f"{v:.0e}" if v < 0.01 or v > 100 else f"{v:.3f}",
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts["C"] = best_val
    return opts, max_acc, best_f1


def _optimize_fi(X_train, y_train, X_test, y_true, opts):
    """Optimize fit_intercept"""
    max_acc = -np.inf
    best_f1 = 0.0
    variable_array = [True, False]
    best_val = variable_array[0]

    with tqdm(variable_array, desc="Optimizing Fit Intercept", leave=False) as pbar:
        for v in pbar:
            test_opts = opts.copy()
            test_opts["fit_intercept"] = v

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, **test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_val = v

            pbar.set_postfix(
                {
                    "fit_int": str(v),
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts["fit_intercept"] = best_val
    return opts, max_acc, best_f1


def _optimize_is(X_train, y_train, X_test, y_true, opts):
    """Optimize intercept_scaling"""
    max_acc = -np.inf
    best_f1 = 0.0
    variable_array = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]  # More focused range
    best_val = variable_array[0]

    with tqdm(variable_array, desc="Optimizing Intercept Scaling", leave=False) as pbar:
        for v in pbar:
            test_opts = opts.copy()
            test_opts["intercept_scaling"] = v

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, **test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_val = v

            pbar.set_postfix(
                {
                    "int_scale": f"{v:.1f}",
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts["intercept_scaling"] = best_val
    return opts, max_acc, best_f1


def _optimize_cw(X_train, y_train, X_test, y_true, opts):
    """Optimize class_weight"""
    max_acc = -np.inf
    best_f1 = 0.0
    variable_array = [None, "balanced"]
    best_val = variable_array[0]

    with tqdm(variable_array, desc="Optimizing Class Weight", leave=False) as pbar:
        for v in pbar:
            test_opts = opts.copy()
            test_opts["class_weight"] = v

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, **test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_val = v

            pbar.set_postfix(
                {
                    "class_wt": str(v) if v is not None else "None",
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts["class_weight"] = best_val
    return opts, max_acc, best_f1


def _optimize_sol(X_train, y_train, X_test, y_true, opts):
    """Optimize solver"""
    max_acc = -np.inf
    best_f1 = 0.0
    variable_array = ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
    # Note: newton-cholesky removed as it's newer and may not be available in all sklearn versions
    best_val = variable_array[0]

    with tqdm(variable_array, desc="Optimizing Solver", leave=False) as pbar:
        for v in pbar:
            test_opts = opts.copy()
            test_opts["solver"] = v

            # Handle solver-penalty compatibility
            if v in ["newton-cg", "lbfgs", "sag"] and opts["penalty"] == "l1":
                test_opts["penalty"] = "l2"  # These solvers don't support L1
            elif v == "liblinear" and opts["penalty"] == "elasticnet":
                test_opts["penalty"] = "l2"  # liblinear doesn't support elasticnet
            elif opts["penalty"] == "elasticnet" and v != "saga":
                pbar.set_postfix(
                    {"solver": v, "status": "skipped (elasticnet needs saga)"}
                )
                continue

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, **test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_val = v

            pbar.set_postfix(
                {
                    "solver": v[:8],
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts["solver"] = best_val
    return opts, max_acc, best_f1


def _optimize_mi(X_train, y_train, X_test, y_true, opts):
    """Optimize max_iter"""
    max_acc = -np.inf
    best_f1 = 0.0
    variable_array = [100, 200, 500, 1000, 2000, 5000]  # More focused range
    best_val = variable_array[0]

    with tqdm(variable_array, desc="Optimizing Max Iterations", leave=False) as pbar:
        for v in pbar:
            test_opts = opts.copy()
            test_opts["max_iter"] = v

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, **test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_val = v

            pbar.set_postfix(
                {
                    "max_iter": v,
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts["max_iter"] = best_val
    return opts, max_acc, best_f1


def _optimize_mc(X_train, y_train, X_test, y_true, opts):
    """Optimize multi_class"""
    max_acc = -np.inf
    best_f1 = 0.0
    variable_array = ["auto", "ovr", "multinomial"]
    best_val = variable_array[0]

    with tqdm(variable_array, desc="Optimizing Multi-Class", leave=False) as pbar:
        for v in pbar:
            test_opts = opts.copy()
            test_opts["multi_class"] = v

            # Handle solver compatibility
            if v == "multinomial" and opts["solver"] == "liblinear":
                test_opts["solver"] = "lbfgs"  # liblinear doesn't support multinomial

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, **test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_val = v

            pbar.set_postfix(
                {
                    "multi_cls": v[:8],
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts["multi_class"] = best_val
    return opts, max_acc, best_f1


def _optimize_l1r(X_train, y_train, X_test, y_true, opts):
    """Optimize l1_ratio (for elasticnet penalty)"""
    max_acc = -np.inf
    best_f1 = 0.0

    # Only optimize if penalty is elasticnet
    if opts["penalty"] != "elasticnet":
        return opts, max_acc, best_f1

    variable_array = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    best_val = variable_array[0]

    with tqdm(variable_array, desc="Optimizing L1 Ratio", leave=False) as pbar:
        for v in pbar:
            test_opts = opts.copy()
            test_opts["l1_ratio"] = v

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, **test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_val = v

            pbar.set_postfix(
                {
                    "l1_ratio": f"{v:.1f}",
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts["l1_ratio"] = best_val
    return opts, max_acc, best_f1


def _optimize_logistic_regression(X_train, y_train, X_test, y_true, cycles=2):
    """
    Optimizes the hyperparameters for LogisticRegression.
    :param X_train: Training data
    :param y_train: Training labels
    :param X_test: Test data
    :param y_true: True labels for the test data
    :param cycles: Number of optimization cycles
    """

    # Define initial parameters
    opts = {
        "penalty": "l2",
        "dual": False,
        "tol": 1e-4,
        "C": 1.0,
        "fit_intercept": True,
        "intercept_scaling": 1.0,
        "class_weight": None,
        "random_state": 42,
        "solver": "lbfgs",
        "max_iter": 1000,
        "verbose": 0,
        "warm_start": False,
        "n_jobs": -1,
        "l1_ratio": None,
        "multi_class": "auto",
    }

    # Optimize hyperparameters
    ma_vec = []
    f1_vec = []

    # Main optimization loop with overall progress bar
    with tqdm(range(cycles), desc="Optimization Cycles", position=0) as cycle_pbar:
        for c in cycle_pbar:
            cycle_pbar.set_description(f"Logistic Regression Cycle {c + 1}/{cycles}")

            opts, _, _ = _optimize_rs(X_train, y_train, X_test, y_true, opts)
            opts, _, _ = _optimize_penalty(X_train, y_train, X_test, y_true, opts)
            opts, _, _ = _optimize_tol(X_train, y_train, X_test, y_true, opts)
            opts, _, _ = _optimize_c(X_train, y_train, X_test, y_true, opts)
            opts, _, _ = _optimize_fi(X_train, y_train, X_test, y_true, opts)
            opts, _, _ = _optimize_is(X_train, y_train, X_test, y_true, opts)
            opts, _, _ = _optimize_cw(X_train, y_train, X_test, y_true, opts)
            opts, _, _ = _optimize_sol(X_train, y_train, X_test, y_true, opts)
            opts, _, _ = _optimize_mi(X_train, y_train, X_test, y_true, opts)
            opts, _, _ = _optimize_mc(X_train, y_train, X_test, y_true, opts)
            opts, ma, f1 = _optimize_l1r(X_train, y_train, X_test, y_true, opts)

            ma_vec.append(ma)
            f1_vec.append(f1)

            cycle_pbar.set_postfix(
                {
                    "accuracy": f"{ma:.4f}",
                    "f1": f"{f1:.4f}",
                    "best_overall": f"{max(ma_vec):.4f}",
                    "penalty": str(opts["penalty"])[:8],
                    "solver": opts["solver"][:8],
                    "C": (
                        f'{opts["C"]:.2e}'
                        if opts["C"] < 0.01 or opts["C"] > 100
                        else f'{opts["C"]:.3f}'
                    ),
                }
            )

    return opts, ma, f1, ma_vec, f1_vec
