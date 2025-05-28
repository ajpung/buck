from typing import Any
import warnings
import numpy as np
from tqdm.auto import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.exceptions import ConvergenceWarning
import time

# Suppress convergence warnings for cleaner progress bars
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Global variables for efficiency
_start_time = None
_max_runtime_per_param = 180  # 3 minutes max per parameter
_min_accuracy_threshold = 0.25  # Stop if accuracy is consistently terrible
_consecutive_failures = 0
_max_consecutive_failures = 3


def _safe_evaluate_model(X_train, y_train, X_test, y_true, **kwargs):
    """Safely evaluate a model configuration"""
    global _consecutive_failures

    try:
        # Use fewer estimators early to speed up evaluation
        test_kwargs = kwargs.copy()
        if "n_estimators" in test_kwargs and test_kwargs["n_estimators"] > 50:
            # Scale down estimators for faster evaluation during optimization
            test_kwargs["n_estimators"] = min(test_kwargs["n_estimators"], 25)

        classifier = RandomForestClassifier(**test_kwargs)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        # Track consecutive failures for early stopping
        if accuracy < _min_accuracy_threshold:
            _consecutive_failures += 1
        else:
            _consecutive_failures = 0

        return accuracy, f1, True
    except Exception:
        _consecutive_failures += 1
        return 0.0, 0.0, False


def _optimize_rs(X_train, y_train, X_test, y_true, opts):
    """Optimize random state"""
    global _start_time, _consecutive_failures
    _start_time = time.time()
    _consecutive_failures = 0

    max_acc = -np.inf
    best_f1 = 0.0
    variable_array = np.arange(
        150
    )  # Reduced from 30 to 5 - random state doesn't need extensive search
    best_val = variable_array[0]

    with tqdm(variable_array, desc="Optimizing Random State", leave=False) as pbar:
        for v in pbar:
            # Early stopping conditions
            if time.time() - _start_time > _max_runtime_per_param:
                pbar.set_description("Random State (TIME LIMIT)")
                break
            if _consecutive_failures >= _max_consecutive_failures:
                pbar.set_description("Random State (POOR ACCURACY)")
                break

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


def _optimize_n_estimators(X_train, y_train, X_test, y_true, opts):
    """Optimize number of estimators"""
    global _start_time, _consecutive_failures
    _start_time = time.time()
    _consecutive_failures = 0

    max_acc = -np.inf
    best_f1 = 0.0
    # Smarter estimator selection based on dataset size
    n_samples = X_train.shape[0]
    if n_samples < 1000:
        variable_array = [5, 10, 25, 50]  # Smaller datasets don't need many trees
    else:
        variable_array = [
            10,
            25,
            50,
            100,
            200,
        ]  # Larger datasets can benefit from more trees
    best_val = variable_array[0]

    with tqdm(variable_array, desc="Optimizing N Estimators", leave=False) as pbar:
        for v in pbar:
            # Early stopping conditions
            if time.time() - _start_time > _max_runtime_per_param:
                pbar.set_description("N Estimators (TIME LIMIT)")
                break
            if _consecutive_failures >= _max_consecutive_failures:
                pbar.set_description("N Estimators (POOR ACCURACY)")
                break

            test_opts = opts.copy()
            test_opts["n_estimators"] = v

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, **test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_val = v

            pbar.set_postfix(
                {
                    "n_est": v,
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts["n_estimators"] = best_val
    return opts, max_acc, best_f1


def _optimize_max_depth(X_train, y_train, X_test, y_true, opts):
    """Optimize max depth"""
    global _start_time, _consecutive_failures
    _start_time = time.time()
    _consecutive_failures = 0

    max_acc = -np.inf
    best_f1 = 0.0
    # Smarter depth selection
    variable_array = [3, 5, 10, 15, None]  # Reduced and more strategic
    best_val = variable_array[0]

    with tqdm(variable_array, desc="Optimizing Max Depth", leave=False) as pbar:
        for v in pbar:
            # Early stopping conditions
            if time.time() - _start_time > _max_runtime_per_param:
                pbar.set_description("Max Depth (TIME LIMIT)")
                break
            if _consecutive_failures >= _max_consecutive_failures:
                pbar.set_description("Max Depth (POOR ACCURACY)")
                break

            test_opts = opts.copy()
            test_opts["max_depth"] = v

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, **test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_val = v

            pbar.set_postfix(
                {
                    "max_depth": str(v) if v is not None else "None",
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts["max_depth"] = best_val
    return opts, max_acc, best_f1


def _optimize_max_features(X_train, y_train, X_test, y_true, opts):
    """Optimize max features"""
    global _start_time, _consecutive_failures
    _start_time = time.time()
    _consecutive_failures = 0

    max_acc = -np.inf
    best_f1 = 0.0
    n_features = X_train.shape[1]

    # Much more efficient feature selection
    if n_features <= 10:
        variable_array = [
            "sqrt",
            "log2",
            None,
        ]  # Skip specific numbers for small datasets
    elif n_features <= 50:
        variable_array = [5, "sqrt", "log2", None]
    else:
        variable_array = ["sqrt", "log2", n_features // 4, None]

    best_val = variable_array[0]

    with tqdm(variable_array, desc="Optimizing Max Features", leave=False) as pbar:
        for v in pbar:
            # Early stopping conditions
            if time.time() - _start_time > _max_runtime_per_param:
                pbar.set_description("Max Features (TIME LIMIT)")
                break
            if _consecutive_failures >= _max_consecutive_failures:
                pbar.set_description("Max Features (POOR ACCURACY)")
                break

            test_opts = opts.copy()
            test_opts["max_features"] = v

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, **test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_val = v

            pbar.set_postfix(
                {
                    "max_feat": str(v) if not isinstance(v, int) else f"{v}",
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts["max_features"] = best_val
    return opts, max_acc, best_f1


def _optimize_criterion(X_train, y_train, X_test, y_true, opts):
    """Optimize splitting criterion"""
    global _start_time, _consecutive_failures
    _start_time = time.time()
    _consecutive_failures = 0

    max_acc = -np.inf
    best_f1 = 0.0
    variable_array = ["gini", "entropy"]  # Removed log_loss for speed
    best_val = variable_array[0]

    with tqdm(variable_array, desc="Optimizing Criterion", leave=False) as pbar:
        for v in pbar:
            # Early stopping conditions
            if time.time() - _start_time > _max_runtime_per_param:
                pbar.set_description("Criterion (TIME LIMIT)")
                break
            if _consecutive_failures >= _max_consecutive_failures:
                pbar.set_description("Criterion (POOR ACCURACY)")
                break

            test_opts = opts.copy()
            test_opts["criterion"] = v

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, **test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_val = v

            pbar.set_postfix(
                {
                    "criterion": v[:8],
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts["criterion"] = best_val
    return opts, max_acc, best_f1


def _optimize_min_samples_config(X_train, y_train, X_test, y_true, opts):
    """Optimize min_samples_split and min_samples_leaf together"""
    global _start_time, _consecutive_failures
    _start_time = time.time()
    _consecutive_failures = 0

    max_acc = -np.inf
    best_f1 = 0.0

    # Reduced configurations for efficiency
    configs = [
        {"min_samples_split": 2, "min_samples_leaf": 1},
        {"min_samples_split": 5, "min_samples_leaf": 1},
        {"min_samples_split": 10, "min_samples_leaf": 2},
        {"min_samples_split": 20, "min_samples_leaf": 5},
    ]
    best_config = configs[0]

    with tqdm(configs, desc="Optimizing Min Samples Config", leave=False) as pbar:
        for config in pbar:
            # Early stopping conditions
            if time.time() - _start_time > _max_runtime_per_param:
                pbar.set_description("Min Samples (TIME LIMIT)")
                break
            if _consecutive_failures >= _max_consecutive_failures:
                pbar.set_description("Min Samples (POOR ACCURACY)")
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
                    "split": config["min_samples_split"],
                    "leaf": config["min_samples_leaf"],
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts.update(best_config)
    return opts, max_acc, best_f1


def _optimize_regularization_params(X_train, y_train, X_test, y_true, opts):
    """Optimize regularization parameters together"""
    global _start_time, _consecutive_failures
    _start_time = time.time()
    _consecutive_failures = 0

    max_acc = -np.inf
    best_f1 = 0.0

    # Simplified regularization configurations
    configs = [
        {
            "min_weight_fraction_leaf": 0.0,
            "min_impurity_decrease": 0.0,
            "ccp_alpha": 0.0,
        },
        {
            "min_weight_fraction_leaf": 0.01,
            "min_impurity_decrease": 0.0,
            "ccp_alpha": 0.0,
        },
        {
            "min_weight_fraction_leaf": 0.0,
            "min_impurity_decrease": 0.01,
            "ccp_alpha": 0.0,
        },
        {
            "min_weight_fraction_leaf": 0.0,
            "min_impurity_decrease": 0.0,
            "ccp_alpha": 0.01,
        },
    ]
    best_config = configs[0]

    with tqdm(configs, desc="Optimizing Regularization", leave=False) as pbar:
        for config in pbar:
            # Early stopping conditions
            if time.time() - _start_time > _max_runtime_per_param:
                pbar.set_description("Regularization (TIME LIMIT)")
                break
            if _consecutive_failures >= _max_consecutive_failures:
                pbar.set_description("Regularization (POOR ACCURACY)")
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
                    "wt_frac": f"{config['min_weight_fraction_leaf']:.2f}",
                    "imp_dec": f"{config['min_impurity_decrease']:.2f}",
                    "ccp": f"{config['ccp_alpha']:.2f}",
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts.update(best_config)
    return opts, max_acc, best_f1


def _optimize_bootstrap_config(X_train, y_train, X_test, y_true, opts):
    """Optimize bootstrap configuration"""
    global _start_time, _consecutive_failures
    _start_time = time.time()
    _consecutive_failures = 0

    max_acc = -np.inf
    best_f1 = 0.0

    # Simplified bootstrap configurations
    configs = [
        {"bootstrap": True, "oob_score": False, "max_samples": None},
        {"bootstrap": True, "oob_score": True, "max_samples": None},
        {"bootstrap": False, "oob_score": False, "max_samples": None},
    ]
    best_config = configs[0]

    with tqdm(configs, desc="Optimizing Bootstrap Config", leave=False) as pbar:
        for config in pbar:
            # Early stopping conditions
            if time.time() - _start_time > _max_runtime_per_param:
                pbar.set_description("Bootstrap (TIME LIMIT)")
                break
            if _consecutive_failures >= _max_consecutive_failures:
                pbar.set_description("Bootstrap (POOR ACCURACY)")
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
                    "bootstrap": "Y" if config["bootstrap"] else "N",
                    "oob": "Y" if config["oob_score"] else "N",
                    "max_samp": (
                        str(config["max_samples"])
                        if config["max_samples"] is not None
                        else "None"
                    ),
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts.update(best_config)
    return opts, max_acc, best_f1


def _optimize_class_weight(X_train, y_train, X_test, y_true, opts):
    """Optimize class weight"""
    global _start_time, _consecutive_failures
    _start_time = time.time()
    _consecutive_failures = 0

    max_acc = -np.inf
    best_f1 = 0.0
    variable_array = [None, "balanced"]  # Removed balanced_subsample for speed
    best_val = variable_array[0]

    with tqdm(variable_array, desc="Optimizing Class Weight", leave=False) as pbar:
        for v in pbar:
            # Early stopping conditions
            if time.time() - _start_time > _max_runtime_per_param:
                pbar.set_description("Class Weight (TIME LIMIT)")
                break
            if _consecutive_failures >= _max_consecutive_failures:
                pbar.set_description("Class Weight (POOR ACCURACY)")
                break

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


def _optimize_max_leaf_nodes(X_train, y_train, X_test, y_true, opts):
    """Optimize max leaf nodes"""
    global _start_time, _consecutive_failures
    _start_time = time.time()
    _consecutive_failures = 0

    max_acc = -np.inf
    best_f1 = 0.0
    # Simplified leaf node values
    variable_array = [50, 100, 500, None]  # Reduced options
    best_val = variable_array[0]

    with tqdm(variable_array, desc="Optimizing Max Leaf Nodes", leave=False) as pbar:
        for v in pbar:
            # Early stopping conditions
            if time.time() - _start_time > _max_runtime_per_param:
                pbar.set_description("Max Leaf Nodes (TIME LIMIT)")
                break
            if _consecutive_failures >= _max_consecutive_failures:
                pbar.set_description("Max Leaf Nodes (POOR ACCURACY)")
                break

            test_opts = opts.copy()
            test_opts["max_leaf_nodes"] = v

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, **test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_val = v

            pbar.set_postfix(
                {
                    "max_leaf": str(v) if v is not None else "None",
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts["max_leaf_nodes"] = best_val
    return opts, max_acc, best_f1


def _optimize_random_forest(X_train, y_train, X_test, y_true, cycles=2):
    """
    Optimizes the hyperparameters for RandomForestClassifier.
    :param X_train: Training data
    :param y_train: Training labels
    :param X_test: Test data
    :param y_true: True labels for the test data
    :param cycles: Number of optimization cycles
    """

    # Quick baseline check to catch data issues early
    print("Running baseline check...")
    baseline_rf = RandomForestClassifier(n_estimators=5, random_state=42)
    baseline_rf.fit(X_train, y_train)
    baseline_acc = baseline_rf.score(X_test, y_true)
    print(f"Baseline accuracy (5 trees): {baseline_acc:.4f}")

    if baseline_acc < 0.2:
        print("⚠️  WARNING: Very low baseline accuracy!")
        print(
            "Consider checking data preprocessing, feature engineering, or class balance."
        )
        print("Proceeding with optimization anyway...")

    # Define initial parameters with efficient defaults
    opts = {
        "n_estimators": 10,  # Start small
        "criterion": "gini",
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "min_weight_fraction_leaf": 0.0,
        "max_features": "sqrt",
        "max_leaf_nodes": None,
        "min_impurity_decrease": 0.0,
        "bootstrap": True,
        "oob_score": False,
        "n_jobs": -1,
        "random_state": 42,
        "verbose": 0,
        "warm_start": False,
        "class_weight": None,
        "ccp_alpha": 0.0,
        "max_samples": None,
        "monotonic_cst": None,
    }

    # Track results
    ma_vec = []
    f1_vec = []

    # Main optimization loop with overall progress bar
    with tqdm(
        range(cycles), desc="Random Forest Optimization Cycles", position=0
    ) as cycle_pbar:
        for c in cycle_pbar:
            cycle_start_time = time.time()
            cycle_pbar.set_description(f"Random Forest Cycle {c + 1}/{cycles}")

            # Core hyperparameters (most impactful for Random Forest)
            opts, _, _ = _optimize_rs(X_train, y_train, X_test, y_true, opts)
            opts, _, _ = _optimize_n_estimators(X_train, y_train, X_test, y_true, opts)
            opts, _, _ = _optimize_max_depth(X_train, y_train, X_test, y_true, opts)
            opts, _, _ = _optimize_max_features(X_train, y_train, X_test, y_true, opts)

            # Tree structure parameters
            opts, _, _ = _optimize_criterion(X_train, y_train, X_test, y_true, opts)
            opts, _, _ = _optimize_min_samples_config(
                X_train, y_train, X_test, y_true, opts
            )
            opts, _, _ = _optimize_max_leaf_nodes(
                X_train, y_train, X_test, y_true, opts
            )

            # Regularization and sampling
            opts, _, _ = _optimize_regularization_params(
                X_train, y_train, X_test, y_true, opts
            )
            opts, _, _ = _optimize_bootstrap_config(
                X_train, y_train, X_test, y_true, opts
            )
            opts, ma, f1 = _optimize_class_weight(
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
                    "n_est": opts["n_estimators"],
                    "depth": (
                        str(opts["max_depth"])
                        if opts["max_depth"] is not None
                        else "None"
                    ),
                    "features": str(opts["max_features"])[:6],
                    "criterion": opts["criterion"][:4],
                    "bootstrap": "Y" if opts["bootstrap"] else "N",
                }
            )

    return opts, ma, f1, ma_vec, f1_vec
