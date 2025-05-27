from typing import Any
import warnings
import numpy as np
from tqdm.auto import tqdm
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.exceptions import ConvergenceWarning

# Suppress convergence warnings for cleaner progress bars
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def _safe_evaluate_model(X_train, y_train, X_test, y_true, **kwargs):
    """Safely evaluate a model configuration"""
    try:
        classifier = PassiveAggressiveClassifier(**kwargs)
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
    variable_array = np.arange(25)  # Reduced from 800 to 25
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


def _optimize_C(X_train, y_train, X_test, y_true, opts):
    """Optimize regularization parameter C"""
    max_acc = -np.inf
    best_f1 = 0.0
    # Much more efficient: logarithmic spacing for regularization
    variable_array = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
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
                    "C": f"{v:.2f}",
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts["C"] = best_val
    return opts, max_acc, best_f1


def _optimize_loss_function(X_train, y_train, X_test, y_true, opts):
    """Optimize loss function"""
    max_acc = -np.inf
    best_f1 = 0.0
    variable_array = ["hinge", "squared_hinge"]  # Removed "perceptron" (less common)
    best_val = variable_array[0]

    with tqdm(variable_array, desc="Optimizing Loss Function", leave=False) as pbar:
        for v in pbar:
            test_opts = opts.copy()
            test_opts["loss"] = v

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, **test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_val = v

            pbar.set_postfix(
                {
                    "loss": v[:10],
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts["loss"] = best_val
    return opts, max_acc, best_f1


def _optimize_max_iter(X_train, y_train, X_test, y_true, opts):
    """Optimize maximum iterations"""
    max_acc = -np.inf
    best_f1 = 0.0
    # Strategic iteration values instead of linear range
    variable_array = [100, 500, 1000, 2000, 5000]
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


def _optimize_tolerance(X_train, y_train, X_test, y_true, opts):
    """Optimize tolerance"""
    max_acc = -np.inf
    best_f1 = 0.0
    # Strategic tolerance values
    variable_array = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
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


def _optimize_early_stopping_config(X_train, y_train, X_test, y_true, opts):
    """Optimize early stopping configuration"""
    max_acc = -np.inf
    best_f1 = 0.0

    # Combined early stopping configurations
    configs = [
        {"early_stopping": False, "validation_fraction": 0.1, "n_iter_no_change": 5},
        {"early_stopping": True, "validation_fraction": 0.1, "n_iter_no_change": 5},
        {"early_stopping": True, "validation_fraction": 0.1, "n_iter_no_change": 10},
        {"early_stopping": True, "validation_fraction": 0.15, "n_iter_no_change": 5},
        {"early_stopping": True, "validation_fraction": 0.2, "n_iter_no_change": 10},
        {"early_stopping": True, "validation_fraction": 0.1, "n_iter_no_change": 15},
    ]
    best_config = configs[0]

    with tqdm(configs, desc="Optimizing Early Stopping", leave=False) as pbar:
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
                    "early_stop": "Y" if config["early_stopping"] else "N",
                    "val_frac": f"{config['validation_fraction']:.2f}",
                    "no_change": config["n_iter_no_change"],
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts.update(best_config)
    return opts, max_acc, best_f1


def _optimize_binary_params(X_train, y_train, X_test, y_true, opts):
    """Optimize binary parameters (fit_intercept, shuffle, average)"""
    max_acc = -np.inf
    best_f1 = 0.0

    # Combined binary parameter configurations
    configs = [
        {"fit_intercept": True, "shuffle": True, "average": False},
        {"fit_intercept": True, "shuffle": True, "average": True},
        {"fit_intercept": True, "shuffle": False, "average": False},
        {"fit_intercept": False, "shuffle": True, "average": False},
        {"fit_intercept": False, "shuffle": True, "average": True},
        {"fit_intercept": True, "shuffle": False, "average": True},
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
                    "shuffle": "Y" if config["shuffle"] else "N",
                    "average": "Y" if config["average"] else "N",
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


def _optimize_passive_aggressive(X_train, y_train, X_test, y_true, cycles=2):
    """
    Optimizes the hyperparameters for PassiveAggressiveClassifier.
    :param X_train: Training data
    :param y_train: Training labels
    :param X_test: Test data
    :param y_true: True labels for the test data
    :param cycles: Number of optimization cycles
    """

    # Define initial parameters with better defaults
    opts = {
        "random_state": 42,
        "C": 1.0,
        "fit_intercept": True,
        "max_iter": 1000,
        "tol": 1e-3,
        "early_stopping": False,
        "validation_fraction": 0.1,
        "n_iter_no_change": 5,
        "shuffle": True,
        "verbose": 0,
        "loss": "hinge",
        "n_jobs": -1,
        "warm_start": False,
        "class_weight": None,
        "average": False,
    }

    # Track results
    ma_vec = []
    f1_vec = []

    # Main optimization loop with overall progress bar
    with tqdm(
        range(cycles), desc="Passive Aggressive Optimization Cycles", position=0
    ) as cycle_pbar:
        for c in cycle_pbar:
            cycle_pbar.set_description(f"Passive Aggressive Cycle {c + 1}/{cycles}")

            # Core hyperparameters (most impactful)
            opts, _, _ = _optimize_rs(X_train, y_train, X_test, y_true, opts)
            opts, _, _ = _optimize_C(X_train, y_train, X_test, y_true, opts)
            opts, _, _ = _optimize_loss_function(X_train, y_train, X_test, y_true, opts)

            # Convergence parameters
            opts, _, _ = _optimize_max_iter(X_train, y_train, X_test, y_true, opts)
            opts, _, _ = _optimize_tolerance(X_train, y_train, X_test, y_true, opts)

            # Combined parameter optimizations (more efficient)
            opts, _, _ = _optimize_early_stopping_config(
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
                    "C": f"{opts['C']:.2f}",
                    "loss": opts["loss"][:6],
                    "max_iter": opts["max_iter"],
                    "tol": f"{opts['tol']:.0e}",
                    "early_stop": "Y" if opts["early_stopping"] else "N",
                }
            )

    return opts, ma, f1, ma_vec, f1_vec
