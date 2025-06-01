from typing import Any
import warnings
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score

# Handle LightGBM import with fallback
try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available. Install with: pip install lightgbm")

# Suppress warnings for cleaner progress bars
warnings.filterwarnings("ignore")


def _safe_evaluate_model(X_train, y_train, X_test, y_true, **kwargs):
    """Safely evaluate a LightGBM model configuration"""
    if not LIGHTGBM_AVAILABLE:
        return 0.0, 0.0, False

    try:
        # LightGBM specific parameters
        lgb_params = kwargs.copy()
        lgb_params["verbose"] = -1  # Suppress output
        lgb_params["force_col_wise"] = True  # Avoid data structure warnings

        classifier = lgb.LGBMClassifier(**lgb_params)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        return accuracy, f1, True
    except Exception:
        return 0.0, 0.0, False


def _optimize_boosting_config(X_train, y_train, X_test, y_true, opts):
    """Optimize boosting configuration (n_estimators + learning_rate together)"""
    max_acc = -np.inf
    best_f1 = 0.0

    # Strategic boosting configurations for LightGBM
    configs = [
        {"n_estimators": 100, "learning_rate": 0.1},  # Default balanced
        {"n_estimators": 200, "learning_rate": 0.05},  # More trees, slower learning
        {"n_estimators": 300, "learning_rate": 0.03},  # Many trees, very slow learning
        {"n_estimators": 500, "learning_rate": 0.01},  # Many trees, very conservative
        {"n_estimators": 150, "learning_rate": 0.08},  # Moderate compromise
        {"n_estimators": 50, "learning_rate": 0.2},  # Few trees, aggressive learning
        {"n_estimators": 75, "learning_rate": 0.15},  # Balanced aggressive
        {"n_estimators": 400, "learning_rate": 0.02},  # Conservative approach
    ]
    best_config = configs[0]

    with tqdm(configs, desc="Optimizing Boosting Config", leave=False) as pbar:
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
                    "n_est": config["n_estimators"],
                    "lr": f"{config['learning_rate']:.3f}",
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts.update(best_config)
    return opts, max_acc, best_f1


def _optimize_tree_structure(X_train, y_train, X_test, y_true, opts):
    """Optimize tree structure parameters together"""
    max_acc = -np.inf
    best_f1 = 0.0

    # Strategic tree structure configurations for LightGBM
    configs = [
        {"max_depth": -1, "num_leaves": 31, "min_child_samples": 20},  # Default
        {"max_depth": 6, "num_leaves": 63, "min_child_samples": 20},  # Controlled depth
        {"max_depth": 8, "num_leaves": 127, "min_child_samples": 20},  # Deeper trees
        {"max_depth": -1, "num_leaves": 15, "min_child_samples": 20},  # Fewer leaves
        {"max_depth": -1, "num_leaves": 63, "min_child_samples": 20},  # More leaves
        {
            "max_depth": -1,
            "num_leaves": 31,
            "min_child_samples": 50,
        },  # Conservative splits
        {"max_depth": 10, "num_leaves": 255, "min_child_samples": 10},  # Complex trees
        {"max_depth": 4, "num_leaves": 15, "min_child_samples": 100},  # Simple trees
    ]
    best_config = configs[0]

    with tqdm(configs, desc="Optimizing Tree Structure", leave=False) as pbar:
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

            depth_str = (
                str(config["max_depth"]) if config["max_depth"] != -1 else "Auto"
            )
            pbar.set_postfix(
                {
                    "depth": depth_str,
                    "leaves": config["num_leaves"],
                    "min_child": config["min_child_samples"],
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts.update(best_config)
    return opts, max_acc, best_f1


def _optimize_regularization(X_train, y_train, X_test, y_true, opts):
    """Optimize regularization parameters together"""
    max_acc = -np.inf
    best_f1 = 0.0

    # Strategic regularization configurations for LightGBM
    configs = [
        {"reg_alpha": 0.0, "reg_lambda": 0.0},  # No regularization
        {"reg_alpha": 0.1, "reg_lambda": 0.1},  # Light regularization
        {"reg_alpha": 1.0, "reg_lambda": 1.0},  # Moderate regularization
        {"reg_alpha": 0.0, "reg_lambda": 1.0},  # L2 only
        {"reg_alpha": 1.0, "reg_lambda": 0.0},  # L1 only (sparsity)
        {"reg_alpha": 0.5, "reg_lambda": 0.5},  # Balanced light
        {"reg_alpha": 2.0, "reg_lambda": 2.0},  # Strong regularization
        {"reg_alpha": 0.1, "reg_lambda": 1.0},  # Light L1 + moderate L2
    ]
    best_config = configs[0]

    with tqdm(configs, desc="Optimizing Regularization", leave=False) as pbar:
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
                    "alpha": f"{config['reg_alpha']:.1f}",
                    "lambda": f"{config['reg_lambda']:.1f}",
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts.update(best_config)
    return opts, max_acc, best_f1


def _optimize_sampling_config(X_train, y_train, X_test, y_true, opts):
    """Optimize sampling configuration"""
    max_acc = -np.inf
    best_f1 = 0.0

    # Strategic sampling configurations for LightGBM
    configs = [
        {"subsample": 1.0, "colsample_bytree": 1.0, "subsample_freq": 0},  # No sampling
        {
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "subsample_freq": 1,
        },  # Moderate sampling
        {
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "subsample_freq": 1,
        },  # Light sampling
        {
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "subsample_freq": 1,
        },  # Aggressive sampling
        {
            "subsample": 0.8,
            "colsample_bytree": 1.0,
            "subsample_freq": 1,
        },  # Row sampling only
        {
            "subsample": 1.0,
            "colsample_bytree": 0.8,
            "subsample_freq": 0,
        },  # Column sampling only
        {
            "subsample": 0.6,
            "colsample_bytree": 0.8,
            "subsample_freq": 1,
        },  # Heavy row sampling
        {
            "subsample": 0.8,
            "colsample_bytree": 0.6,
            "subsample_freq": 0,
        },  # Heavy column sampling
    ]
    best_config = configs[0]

    with tqdm(configs, desc="Optimizing Sampling Config", leave=False) as pbar:
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
                    "subsample": f"{config['subsample']:.1f}",
                    "colsample": f"{config['colsample_bytree']:.1f}",
                    "freq": config["subsample_freq"],
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts.update(best_config)
    return opts, max_acc, best_f1


def _optimize_objective_metric(X_train, y_train, X_test, y_true, opts):
    """Optimize objective function and metric"""
    max_acc = -np.inf
    best_f1 = 0.0

    # Determine if binary or multi-class
    n_classes = len(np.unique(y_train))

    if n_classes == 2:  # Binary classification
        configs = [
            {"objective": "binary", "metric": "binary_logloss"},
            {"objective": "binary", "metric": "binary_error"},
        ]
    else:  # Multi-class classification
        configs = [
            {"objective": "multiclass", "metric": "multi_logloss"},
            {"objective": "multiclass", "metric": "multi_error"},
        ]

    best_config = configs[0]

    with tqdm(configs, desc="Optimizing Objective/Metric", leave=False) as pbar:
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
                    "objective": config["objective"][:6],
                    "metric": config["metric"][:8],
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts.update(best_config)
    return opts, max_acc, best_f1


def _optimize_advanced_params(X_train, y_train, X_test, y_true, opts):
    """Optimize advanced LightGBM parameters"""
    max_acc = -np.inf
    best_f1 = 0.0

    # Advanced parameter configurations for LightGBM
    configs = [
        {
            "boosting_type": "gbdt",
            "min_gain_to_split": 0.0,
            "min_sum_hessian_in_leaf": 1e-3,
        },  # Default
        {
            "boosting_type": "dart",
            "min_gain_to_split": 0.0,
            "min_sum_hessian_in_leaf": 1e-3,
        },  # DART boosting
        {
            "boosting_type": "gbdt",
            "min_gain_to_split": 0.1,
            "min_sum_hessian_in_leaf": 1e-3,
        },  # Conservative splits
        {
            "boosting_type": "gbdt",
            "min_gain_to_split": 0.0,
            "min_sum_hessian_in_leaf": 1e-2,
        },  # Conservative hessian
        {
            "boosting_type": "goss",
            "min_gain_to_split": 0.0,
            "min_sum_hessian_in_leaf": 1e-3,
        },  # GOSS boosting
        {
            "boosting_type": "gbdt",
            "min_gain_to_split": 0.05,
            "min_sum_hessian_in_leaf": 5e-3,
        },  # Balanced conservative
    ]

    best_config = configs[0]

    with tqdm(configs, desc="Optimizing Advanced Params", leave=False) as pbar:
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
                    "boost": config["boosting_type"][:4],
                    "min_gain": f"{config['min_gain_to_split']:.2f}",
                    "hessian": f"{config['min_sum_hessian_in_leaf']:.0e}",
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts.update(best_config)
    return opts, max_acc, best_f1


def _optimize_lightgbm(X_train, y_train, X_test, y_true, cycles=2):
    """
    Optimizes the hyperparameters for LightGBM Classifier.
    :param X_train: Training data
    :param y_train: Training labels
    :param X_test: Test data
    :param y_true: True labels for the test data
    :param cycles: Number of optimization cycles
    """

    if not LIGHTGBM_AVAILABLE:
        print("LightGBM is not available. Please install it with: pip install lightgbm")
        return {}, 0.0, 0.0, [], []

    # Determine problem type
    n_classes = len(np.unique(y_train))

    # Define initial parameters with good defaults
    opts = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": -1,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "subsample_freq": 0,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "min_gain_to_split": 0.0,
        "min_sum_hessian_in_leaf": 1e-3,
        "boosting_type": "gbdt",
        "objective": "multiclass" if n_classes > 2 else "binary",
        "metric": "multi_logloss" if n_classes > 2 else "binary_logloss",
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
        "force_col_wise": True,
    }

    # Track results
    ma_vec = []
    f1_vec = []

    # Main optimization loop with overall progress bar
    with tqdm(
        range(cycles), desc="LightGBM Optimization Cycles", position=0
    ) as cycle_pbar:
        for c in cycle_pbar:
            cycle_pbar.set_description(f"LightGBM Cycle {c + 1}/{cycles}")

            # Core hyperparameters (most impactful for LightGBM)
            opts, _, _ = _optimize_boosting_config(
                X_train, y_train, X_test, y_true, opts
            )
            opts, _, _ = _optimize_tree_structure(
                X_train, y_train, X_test, y_true, opts
            )
            opts, _, _ = _optimize_regularization(
                X_train, y_train, X_test, y_true, opts
            )
            opts, _, _ = _optimize_sampling_config(
                X_train, y_train, X_test, y_true, opts
            )
            opts, _, _ = _optimize_objective_metric(
                X_train, y_train, X_test, y_true, opts
            )
            opts, ma, f1 = _optimize_advanced_params(
                X_train, y_train, X_test, y_true, opts
            )

            ma_vec.append(ma)
            f1_vec.append(f1)

            cycle_pbar.set_postfix(
                {
                    "accuracy": f"{ma:.4f}",
                    "f1": f"{f1:.4f}",
                    "best_overall": f"{max(ma_vec):.4f}",
                    "n_est": opts["n_estimators"],
                    "lr": f"{opts['learning_rate']:.3f}",
                    "leaves": opts["num_leaves"],
                    "reg_alpha": f"{opts['reg_alpha']:.1f}",
                    "reg_lambda": f"{opts['reg_lambda']:.1f}",
                    "subsample": f"{opts['subsample']:.1f}",
                    "boost_type": opts["boosting_type"][:4],
                }
            )

    return opts, ma, f1, ma_vec, f1_vec
