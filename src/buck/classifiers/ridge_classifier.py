from typing import Any
import warnings
import numpy as np
from tqdm.auto import tqdm
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.exceptions import ConvergenceWarning

# Suppress convergence warnings for cleaner progress bars
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def _safe_evaluate_model(X_train, y_train, X_test, y_true, **kwargs):
    """Safely evaluate a model configuration"""
    try:
        classifier = RidgeClassifier(**kwargs)
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
    variable_array = np.arange(20)  # Reduced from 800 to 20
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


def _optimize_alpha(X_train, y_train, X_test, y_true, opts):
    """Optimize regularization parameter alpha"""
    max_acc = -np.inf
    best_f1 = 0.0
    # Logarithmic spacing for regularization parameter
    variable_array = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
    best_val = variable_array[0]

    with tqdm(variable_array, desc="Optimizing Alpha", leave=False) as pbar:
        for v in pbar:
            test_opts = opts.copy()
            test_opts["alpha"] = v

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, **test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_val = v

            pbar.set_postfix(
                {
                    "alpha": f"{v:.3f}",
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts["alpha"] = best_val
    return opts, max_acc, best_f1


def _optimize_solver(X_train, y_train, X_test, y_true, opts):
    """Optimize solver algorithm"""
    max_acc = -np.inf
    best_f1 = 0.0

    # Efficient solver selection based on data size
    n_samples, n_features = X_train.shape

    if n_samples > 10000 and n_features > 1000:
        # Large datasets: focus on scalable solvers
        variable_array = ["saga", "sag", "lbfgs", "auto"]
    elif n_samples < 1000:
        # Small datasets: all solvers work well
        variable_array = ["auto", "lsqr", "svd", "cholesky", "lbfgs"]
    else:
        # Medium datasets: balanced selection
        variable_array = ["auto", "lsqr", "sag", "saga", "lbfgs", "cholesky"]

    best_val = variable_array[0]

    with tqdm(variable_array, desc="Optimizing Solver", leave=False) as pbar:
        for v in pbar:
            test_opts = opts.copy()
            test_opts["solver"] = v

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


def _optimize_convergence_config(X_train, y_train, X_test, y_true, opts):
    """Optimize convergence configuration (max_iter + tol together)"""
    max_acc = -np.inf
    best_f1 = 0.0

    # Combined convergence configurations for efficiency
    configs = [
        {"max_iter": None, "tol": 1e-3},  # Default unlimited iterations
        {"max_iter": 100, "tol": 1e-3},
        {"max_iter": 500, "tol": 1e-3},
        {"max_iter": 1000, "tol": 1e-3},
        {"max_iter": 2000, "tol": 1e-3},
        {"max_iter": 1000, "tol": 1e-4},  # Tighter tolerance
        {"max_iter": 1000, "tol": 1e-5},  # Very tight tolerance
        {"max_iter": 5000, "tol": 1e-4},  # More iterations + tight tolerance
    ]
    best_config = configs[0]

    with tqdm(configs, desc="Optimizing Convergence", leave=False) as pbar:
        for config in pbar:
            test_opts = opts.copy()
            test_opts.update(config)

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, **test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_config = config

            max_iter_str = (
                str(config["max_iter"]) if config["max_iter"] is not None else "None"
            )
            pbar.set_postfix(
                {
                    "max_iter": max_iter_str[:6],
                    "tol": f"{config['tol']:.0e}",
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts.update(best_config)
    return opts, max_acc, best_f1


def _optimize_binary_params(X_train, y_train, X_test, y_true, opts):
    """Optimize binary parameters together"""
    max_acc = -np.inf
    best_f1 = 0.0

    # Combined binary parameter configurations
    configs = [
        {"fit_intercept": True, "copy_X": True, "positive": False},
        {"fit_intercept": True, "copy_X": False, "positive": False},
        {"fit_intercept": False, "copy_X": True, "positive": False},
        {"fit_intercept": False, "copy_X": False, "positive": False},
        {"fit_intercept": True, "copy_X": True, "positive": True},  # Positive weights
        {"fit_intercept": False, "copy_X": True, "positive": True},
    ]
    best_config = configs[0]

    with tqdm(configs, desc="Optimizing Binary Parameters", leave=False) as pbar:
        for config in pbar:
            test_opts = opts.copy()
            test_opts.update(config)

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, **test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_config = config

            pbar.set_postfix(
                {
                    "intercept": "Y" if config["fit_intercept"] else "N",
                    "copy_X": "Y" if config["copy_X"] else "N",
                    "positive": "Y" if config["positive"] else "N",
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts.update(best_config)
    return opts, max_acc, best_f1


def _optimize_class_weight(X_train, y_train, X_test, y_true, opts):
    """Optimize class weight"""
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


def _optimize_ridge(X_train, y_train, X_test, y_true, cycles=2):
    """
    Optimizes the hyperparameters for RidgeClassifier.
    :param X_train: Training data
    :param y_train: Training labels
    :param X_test: Test data
    :param y_true: True labels for the test data
    :param cycles: Number of optimization cycles
    """

    # Define initial parameters with better defaults
    opts = {
        "random_state": 42,
        "alpha": 1.0,
        "fit_intercept": True,
        "copy_X": True,
        "max_iter": None,
        "tol": 1e-3,
        "class_weight": None,
        "solver": "auto",
        "positive": False,
    }

    # Track results
    ma_vec = []
    f1_vec = []

    # Main optimization loop with overall progress bar
    with tqdm(
        range(cycles), desc="Ridge Classifier Optimization Cycles", position=0
    ) as cycle_pbar:
        for c in cycle_pbar:
            cycle_pbar.set_description(f"Ridge Classifier Cycle {c + 1}/{cycles}")

            # Core hyperparameters (most impactful for Ridge)
            opts, _, _ = _optimize_rs(X_train, y_train, X_test, y_true, opts)
            opts, _, _ = _optimize_alpha(X_train, y_train, X_test, y_true, opts)
            opts, _, _ = _optimize_solver(X_train, y_train, X_test, y_true, opts)

            # Combined parameter optimizations (more efficient)
            opts, _, _ = _optimize_convergence_config(
                X_train, y_train, X_test, y_true, opts
            )
            opts, _, _ = _optimize_binary_params(X_train, y_train, X_test, y_true, opts)
            opts, ma, f1 = _optimize_class_weight(
                X_train, y_train, X_test, y_true, opts
            )

            ma_vec.append(ma)
            f1_vec.append(f1)

            cycle_pbar.set_postfix(
                {
                    "accuracy": f"{ma:.4f}",
                    "f1": f"{f1:.4f}",
                    "best_overall": f"{max(ma_vec):.4f}",
                    "alpha": f"{opts['alpha']:.3f}",
                    "solver": opts["solver"][:6],
                    "max_iter": (
                        str(opts["max_iter"])
                        if opts["max_iter"] is not None
                        else "None"
                    ),
                    "tol": f"{opts['tol']:.0e}",
                    "intercept": "Y" if opts["fit_intercept"] else "N",
                }
            )

    return opts, ma, f1, ma_vec, f1_vec
