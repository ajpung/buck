from typing import Any
import warnings
import numpy as np
import time
from tqdm.auto import tqdm
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.exceptions import ConvergenceWarning

# Suppress convergence warnings for cleaner progress bars
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Global efficiency controls
_max_time_per_step = 180  # 3 minutes max per optimization step (SGD is fast)
_max_time_per_model = 900  # 30 seconds max per model evaluation
_min_accuracy_threshold = 0.15  # Stop if accuracy is consistently terrible
_consecutive_failures = 0
_max_consecutive_failures = 3


def _safe_evaluate_model(X_train, y_train, X_test, y_true, **kwargs):
    """Safely evaluate a model configuration with timeout protection"""
    global _consecutive_failures

    try:
        start_time = time.time()
        classifier = SGDClassifier(**kwargs)
        classifier.fit(X_train, y_train)

        # Check if training took too long
        if time.time() - start_time > _max_time_per_model:
            print(f"⏰ SGD timeout after {_max_time_per_model}s")
            return 0.0, 0.0, False

        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        # Track consecutive failures
        if accuracy < _min_accuracy_threshold:
            _consecutive_failures += 1
        else:
            _consecutive_failures = 0

        return accuracy, f1, True
    except Exception as e:
        _consecutive_failures += 1
        return 0.0, 0.0, False


def _get_smart_loss_penalty_combos(n_samples, n_features):
    """Get smart loss-penalty combinations based on dataset characteristics"""

    # Always include the most effective combinations
    combinations = [
        # Most effective for most problems
        {"loss": "hinge", "penalty": "l2"},  # SVM-like, very robust
        {"loss": "log_loss", "penalty": "l2"},  # Logistic regression-like
        {"loss": "modified_huber", "penalty": "l2"},  # Robust to outliers
        # Sparsity-inducing for high-dimensional data
        {"loss": "log_loss", "penalty": "l1"},  # Sparse logistic
        {"loss": "hinge", "penalty": "l1"},  # Sparse SVM
    ]

    # Add elasticnet for high-dimensional data
    if n_features > 50:
        combinations.append({"loss": "log_loss", "penalty": "elasticnet"})

    # Add additional combos for smaller datasets (more exploration time)
    if n_samples < 5000:
        combinations.extend(
            [
                {"loss": "squared_hinge", "penalty": "l2"},
                {"loss": "perceptron", "penalty": "l2"},
            ]
        )

    return combinations


def _optimize_loss_penalty_combo(X_train, y_train, X_test, y_true, opts):
    """Optimize loss function and penalty with dataset-aware selection"""
    global _consecutive_failures
    start_time = time.time()
    _consecutive_failures = 0

    max_acc = -np.inf
    best_f1 = 0.0
    n_samples, n_features = X_train.shape

    # Get smart combinations based on dataset
    combinations = _get_smart_loss_penalty_combos(n_samples, n_features)
    best_combo = combinations[0]

    with tqdm(combinations, desc="Optimizing Loss-Penalty Combo", leave=False) as pbar:
        for combo in pbar:
            # Early stopping conditions
            if time.time() - start_time > _max_time_per_step:
                pbar.set_description("Loss-Penalty (TIME LIMIT)")
                break
            if _consecutive_failures >= _max_consecutive_failures:
                pbar.set_description("Loss-Penalty (POOR ACCURACY)")
                break

            test_opts = opts.copy()
            test_opts.update(combo)

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, **test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_combo = combo

            pbar.set_postfix(
                {
                    "loss": combo["loss"][:6],
                    "penalty": combo["penalty"][:6] if combo["penalty"] else "None",
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts.update(best_combo)
    return opts, max_acc, best_f1


def _optimize_alpha(X_train, y_train, X_test, y_true, opts):
    """Optimize regularization strength with smart range selection"""
    global _consecutive_failures
    start_time = time.time()
    _consecutive_failures = 0

    max_acc = -np.inf
    best_f1 = 0.0

    # Smart alpha selection based on dataset size
    n_samples = X_train.shape[0]
    if n_samples < 1000:
        # Smaller datasets need less regularization
        variable_array = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    elif n_samples < 10000:
        # Medium datasets - balanced range
        variable_array = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    else:
        # Large datasets can handle more regularization
        variable_array = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]

    best_val = variable_array[2]  # Start with middle value

    with tqdm(variable_array, desc="Optimizing Alpha", leave=False) as pbar:
        for v in pbar:
            # Early stopping conditions
            if time.time() - start_time > _max_time_per_step:
                pbar.set_description("Alpha (TIME LIMIT)")
                break
            if _consecutive_failures >= _max_consecutive_failures:
                pbar.set_description("Alpha (POOR ACCURACY)")
                break

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
                    "alpha": f"{v:.0e}",
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts["alpha"] = best_val
    return opts, max_acc, best_f1


def _optimize_learning_rate_config(X_train, y_train, X_test, y_true, opts):
    """Optimize learning rate with most effective configurations"""
    global _consecutive_failures
    start_time = time.time()
    _consecutive_failures = 0

    max_acc = -np.inf
    best_f1 = 0.0

    # Simplified to most effective configurations
    configs = [
        {"learning_rate": "optimal", "eta0": 0.0, "power_t": 0.5},  # Usually best
        {"learning_rate": "constant", "eta0": 0.01, "power_t": 0.5},
        {"learning_rate": "constant", "eta0": 0.1, "power_t": 0.5},
        {"learning_rate": "invscaling", "eta0": 0.01, "power_t": 0.5},
        {"learning_rate": "adaptive", "eta0": 0.01, "power_t": 0.5},
    ]
    best_config = configs[0]

    with tqdm(configs, desc="Optimizing Learning Rate", leave=False) as pbar:
        for config in pbar:
            # Early stopping conditions
            if time.time() - start_time > _max_time_per_step:
                pbar.set_description("Learning Rate (TIME LIMIT)")
                break
            if _consecutive_failures >= _max_consecutive_failures:
                pbar.set_description("Learning Rate (POOR ACCURACY)")
                break

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
                    "lr_sched": config["learning_rate"][:6],
                    "eta0": f"{config['eta0']:.3f}",
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts.update(best_config)
    return opts, max_acc, best_f1


def _optimize_l1_ratio(X_train, y_train, X_test, y_true, opts):
    """Optimize L1 ratio for ElasticNet penalty (if applicable)"""
    global _consecutive_failures
    start_time = time.time()
    _consecutive_failures = 0

    max_acc = -np.inf
    best_f1 = 0.0

    # Only optimize if penalty is elasticnet
    if opts["penalty"] != "elasticnet":
        return opts, max_acc, best_f1

    # Focused L1 ratios - most effective values
    variable_array = [0.15, 0.3, 0.5, 0.7, 0.85]
    best_val = variable_array[2]  # Start with 0.5

    with tqdm(variable_array, desc="Optimizing L1 Ratio", leave=False) as pbar:
        for v in pbar:
            # Early stopping conditions
            if time.time() - start_time > _max_time_per_step:
                pbar.set_description("L1 Ratio (TIME LIMIT)")
                break
            if _consecutive_failures >= _max_consecutive_failures:
                pbar.set_description("L1 Ratio (POOR ACCURACY)")
                break

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
                    "l1_ratio": f"{v:.2f}",
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts["l1_ratio"] = best_val
    return opts, max_acc, best_f1


def _optimize_convergence_config(X_train, y_train, X_test, y_true, opts):
    """Optimize convergence with dataset-aware iteration limits"""
    global _consecutive_failures
    start_time = time.time()
    _consecutive_failures = 0

    max_acc = -np.inf
    best_f1 = 0.0
    n_samples = X_train.shape[0]

    # Smart iteration limits based on dataset size
    if n_samples < 1000:
        max_iter_options = [500, 1000, 2000]
    elif n_samples < 10000:
        max_iter_options = [1000, 2000, 5000]
    else:
        max_iter_options = [2000, 5000, 10000]  # Cap at 10K for large datasets

    # Simplified convergence configurations
    configs = [
        {"max_iter": max_iter_options[0], "tol": 1e-3, "early_stopping": False},
        {"max_iter": max_iter_options[1], "tol": 1e-3, "early_stopping": False},
        {"max_iter": max_iter_options[2], "tol": 1e-3, "early_stopping": False},
        {
            "max_iter": max_iter_options[2],
            "tol": 1e-3,
            "early_stopping": True,
            "validation_fraction": 0.1,
            "n_iter_no_change": 5,
        },
    ]
    best_config = configs[0]

    with tqdm(configs, desc="Optimizing Convergence", leave=False) as pbar:
        for config in pbar:
            # Early stopping conditions
            if time.time() - start_time > _max_time_per_step:
                pbar.set_description("Convergence (TIME LIMIT)")
                break
            if _consecutive_failures >= _max_consecutive_failures:
                pbar.set_description("Convergence (POOR ACCURACY)")
                break

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
                    "max_iter": config["max_iter"],
                    "tol": f"{config['tol']:.0e}",
                    "early_stop": "Y" if config["early_stopping"] else "N",
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts.update(best_config)
    return opts, max_acc, best_f1


def _optimize_final_params(X_train, y_train, X_test, y_true, opts):
    """Optimize final parameters - class weight and key binary settings"""
    global _consecutive_failures
    start_time = time.time()
    _consecutive_failures = 0

    max_acc = -np.inf
    best_f1 = 0.0

    # Most impactful final parameter combinations
    configs = [
        {
            "class_weight": None,
            "fit_intercept": True,
            "shuffle": True,
            "average": False,
        },
        {
            "class_weight": "balanced",
            "fit_intercept": True,
            "shuffle": True,
            "average": False,
        },
        {"class_weight": None, "fit_intercept": True, "shuffle": True, "average": True},
        {
            "class_weight": "balanced",
            "fit_intercept": True,
            "shuffle": True,
            "average": True,
        },
        {
            "class_weight": None,
            "fit_intercept": False,
            "shuffle": True,
            "average": False,
        },
    ]
    best_config = configs[0]

    with tqdm(configs, desc="Optimizing Final Params", leave=False) as pbar:
        for config in pbar:
            # Early stopping conditions
            if time.time() - start_time > _max_time_per_step:
                pbar.set_description("Final Params (TIME LIMIT)")
                break
            if _consecutive_failures >= _max_consecutive_failures:
                pbar.set_description("Final Params (POOR ACCURACY)")
                break

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
                    "class_wt": (
                        "bal" if config["class_weight"] == "balanced" else "none"
                    ),
                    "intercept": "Y" if config["fit_intercept"] else "N",
                    "average": "Y" if config["average"] else "N",
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts.update(best_config)
    return opts, max_acc, best_f1


def _optimize_sgd(X_train, y_train, X_test, y_true, cycles=2):
    """
    FAST optimized hyperparameters for SGDClassifier.
    :param X_train: Training data
    :param y_train: Training labels
    :param X_test: Test data
    :param y_true: True labels for the test data
    :param cycles: Number of optimization cycles
    """

    n_samples, n_features = X_train.shape
    print(f"Dataset: {n_samples} samples, {n_features} features")

    # Quick baseline check
    print("Running baseline check...")
    baseline_sgd = SGDClassifier(random_state=42, max_iter=1000)
    baseline_sgd.fit(X_train, y_train)
    baseline_acc = baseline_sgd.score(X_test, y_true)
    print(f"Baseline SGD accuracy: {baseline_acc:.4f}")

    if baseline_acc < 0.1:
        print("⚠️  WARNING: Very low baseline accuracy!")
        print(
            "Consider checking data scaling (SGD needs standardized features!), class balance, or data quality."
        )

    # Define initial parameters with smart defaults
    opts = {
        "loss": "hinge",
        "penalty": "l2",
        "alpha": 1e-4,
        "l1_ratio": 0.15,
        "fit_intercept": True,
        "max_iter": 1000,
        "tol": 1e-3,
        "shuffle": True,
        "verbose": 0,
        "epsilon": 0.1,
        "n_jobs": -1,
        "random_state": 42,
        "learning_rate": "optimal",
        "eta0": 0.0,
        "power_t": 0.5,
        "early_stopping": False,
        "validation_fraction": 0.1,
        "n_iter_no_change": 5,
        "class_weight": None,
        "warm_start": False,
        "average": False,
    }

    # Track results
    ma_vec = []
    f1_vec = []

    # Main optimization loop
    with tqdm(range(cycles), desc="FAST SGD Optimization", position=0) as cycle_pbar:
        for c in cycle_pbar:
            cycle_start_time = time.time()
            cycle_pbar.set_description(f"SGD Cycle {c + 1}/{cycles}")

            # Core optimizations in order of importance
            opts, _, _ = _optimize_loss_penalty_combo(
                X_train, y_train, X_test, y_true, opts
            )
            opts, _, _ = _optimize_alpha(X_train, y_train, X_test, y_true, opts)
            opts, _, _ = _optimize_learning_rate_config(
                X_train, y_train, X_test, y_true, opts
            )

            # Conditional optimization
            opts, _, _ = _optimize_l1_ratio(X_train, y_train, X_test, y_true, opts)

            # Convergence and final parameters
            opts, _, _ = _optimize_convergence_config(
                X_train, y_train, X_test, y_true, opts
            )
            opts, ma, f1 = _optimize_final_params(
                X_train, y_train, X_test, y_true, opts
            )

            ma_vec.append(ma)
            f1_vec.append(f1)

            cycle_time = time.time() - cycle_start_time

            cycle_pbar.set_postfix(
                {
                    "accuracy": f"{ma:.4f}",
                    "f1": f"{f1:.4f}",
                    "best_overall": f"{max(ma_vec):.4f}",
                    "cycle_time": f"{cycle_time:.1f}s",
                    "loss": opts["loss"][:6],
                    "penalty": opts["penalty"][:6] if opts["penalty"] else "None",
                    "alpha": f"{opts['alpha']:.0e}",
                    "lr_sched": opts["learning_rate"][:6],
                    "max_iter": opts["max_iter"],
                    "baseline_beat": f"{ma - baseline_acc:+.4f}",
                }
            )

    return opts, ma, f1, ma_vec, f1_vec
