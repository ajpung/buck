from typing import Any
import warnings
import numpy as np
import time
from tqdm.auto import tqdm
from sklearn.ensemble import (
    StackingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.exceptions import ConvergenceWarning

# Import XGBoost (your top performer!)
try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    print("⚠️  XGBoost not available - install with: pip install xgboost")
    XGBOOST_AVAILABLE = False

# Suppress convergence warnings for cleaner progress bars
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Global efficiency controls - more aggressive for stacking due to CV overhead
_max_time_per_step = 900  # 10 minutes max per optimization step (stacking is expensive)
_max_time_per_model = 900  # 3 minutes max per model evaluation (CV makes it slow)
_min_accuracy_threshold = 0.15  # Stop if accuracy is consistently terrible
_consecutive_failures = 0
_max_consecutive_failures = 3


def _safe_evaluate_model(
    X_train, y_train, X_test, y_true, estimators, final_estimator, **stacking_params
):
    """Safely evaluate a stacking model configuration with timeout protection"""
    global _consecutive_failures

    try:
        start_time = time.time()
        classifier = StackingClassifier(
            estimators=estimators, final_estimator=final_estimator, **stacking_params
        )
        classifier.fit(X_train, y_train)

        # Check if training took too long (stacking is slow due to CV)
        # if time.time() - start_time > _max_time_per_model:
        #    print(f"⏰ Stacking timeout after {_max_time_per_model}s")
        #    return 0.0, 0.0, False

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


def _create_fast_base_estimator_combinations(n_samples, n_features):
    """Create FAST base estimator combinations focused on your top performers"""

    # Your proven fast performers
    fast_estimators = [
        ("lr", LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1)),
        ("nb", GaussianNB()),
        ("sgd", SGDClassifier(random_state=42, max_iter=1000, n_jobs=-1)),
    ]

    # Your top tree performers with smaller sizes for stacking speed
    if n_samples < 2000:
        tree_estimators = [
            ("rf", RandomForestClassifier(n_estimators=25, random_state=42, n_jobs=-1)),
            ("et", ExtraTreesClassifier(n_estimators=25, random_state=42, n_jobs=-1)),
        ]
    elif n_samples < 10000:
        tree_estimators = [
            ("rf", RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)),
            ("et", ExtraTreesClassifier(n_estimators=50, random_state=42, n_jobs=-1)),
        ]
    else:
        tree_estimators = [
            ("rf", RandomForestClassifier(n_estimators=75, random_state=42, n_jobs=-1)),
        ]  # Only RF for large datasets

    # XGBoost - your #1 performer (optimized for stacking)
    if XGBOOST_AVAILABLE:
        if n_samples < 2000:
            xgb_estimators = [
                (
                    "xgb",
                    XGBClassifier(
                        n_estimators=25, random_state=42, verbosity=0, n_jobs=-1
                    ),
                ),
            ]
        elif n_samples < 10000:
            xgb_estimators = [
                (
                    "xgb",
                    XGBClassifier(
                        n_estimators=50, random_state=42, verbosity=0, n_jobs=-1
                    ),
                ),
            ]
        else:
            xgb_estimators = [
                (
                    "xgb",
                    XGBClassifier(
                        n_estimators=75, random_state=42, verbosity=0, n_jobs=-1
                    ),
                ),
            ]
    else:
        xgb_estimators = []

    # Smart combinations based on dataset size (fewer combos for speed)
    combinations = []

    if XGBOOST_AVAILABLE:
        # Your top performers together
        combinations.append(xgb_estimators + tree_estimators[:2])  # xgb + rf + et
        combinations.append(
            xgb_estimators + tree_estimators[:1] + fast_estimators[:1]
        )  # xgb + rf + lr
        combinations.append(xgb_estimators + fast_estimators[:2])  # xgb + lr + nb

    # Tree-based combinations
    combinations.append(tree_estimators[:2] + fast_estimators[:1])  # rf + et + lr
    combinations.append(tree_estimators[:1] + fast_estimators[:2])  # rf + lr + nb

    # Fast-only for large datasets
    if n_samples > 10000:
        combinations.append(fast_estimators[:3])  # lr + nb + sgd
    else:
        combinations.append(fast_estimators[:2])  # lr + nb

    return combinations


def _optimize_base_estimators(X_train, y_train, X_test, y_true, opts):
    """Optimize base estimator combinations with speed focus"""
    global _consecutive_failures
    start_time = time.time()
    _consecutive_failures = 0

    max_acc = -np.inf
    best_f1 = 0.0
    n_samples, n_features = X_train.shape

    # Get fast estimator combinations
    estimator_combinations = _create_fast_base_estimator_combinations(
        n_samples, n_features
    )
    best_estimators = estimator_combinations[0]

    with tqdm(
        estimator_combinations, desc="Optimizing Base Estimators", leave=False
    ) as pbar:
        for estimators in pbar:
            if _consecutive_failures >= _max_consecutive_failures:
                pbar.set_description("Base Estimators (POOR ACCURACY)")
                break

            # Create lightweight final estimator
            final_estimator = LogisticRegression(
                random_state=42, max_iter=500, n_jobs=-1
            )

            accuracy, f1, success = _safe_evaluate_model(
                X_train,
                y_train,
                X_test,
                y_true,
                estimators,
                final_estimator,
                **opts["stacking"],
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_estimators = estimators

            estimator_names = [name for name, _ in estimators]
            pbar.set_postfix(
                {
                    "combo": "+".join(estimator_names)[:15],
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts["base_estimators"] = best_estimators
    return opts, max_acc, best_f1


def _optimize_meta_estimator(X_train, y_train, X_test, y_true, opts):
    """Optimize meta estimator (final estimator) with fast options"""
    global _consecutive_failures
    start_time = time.time()
    _consecutive_failures = 0

    max_acc = -np.inf
    best_f1 = 0.0
    n_samples = X_train.shape[0]

    # Fast meta estimators only
    meta_estimators = [
        (
            "LogisticRegression",
            LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1),
        ),
        (
            "LogisticRegression_L1",
            LogisticRegression(
                penalty="l1", solver="liblinear", random_state=42, max_iter=1000
            ),
        ),
        ("SGD", SGDClassifier(random_state=42, max_iter=1000, n_jobs=-1)),
    ]

    # Add XGBoost meta-learner for smaller datasets
    if XGBOOST_AVAILABLE and n_samples < 5000:
        meta_estimators.append(
            (
                "XGBoost",
                XGBClassifier(n_estimators=25, random_state=42, verbosity=0, n_jobs=-1),
            )
        )

    # Add lightweight RF for small datasets only
    if n_samples < 2000:
        meta_estimators.append(
            (
                "RandomForest",
                RandomForestClassifier(n_estimators=25, random_state=42, n_jobs=-1),
            )
        )

    best_meta = meta_estimators[0]

    with tqdm(meta_estimators, desc="Optimizing Meta Estimator", leave=False) as pbar:
        for name, meta_estimator in pbar:
            # Early stopping conditions
            if _consecutive_failures >= _max_consecutive_failures:
                pbar.set_description("Meta Estimator (POOR ACCURACY)")
                break

            accuracy, f1, success = _safe_evaluate_model(
                X_train,
                y_train,
                X_test,
                y_true,
                opts["base_estimators"],
                meta_estimator,
                **opts["stacking"],
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_meta = (name, meta_estimator)

            pbar.set_postfix(
                {
                    "meta": name[:12],
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts["meta_estimator"] = best_meta[1]
    opts["meta_name"] = best_meta[0]
    return opts, max_acc, best_f1


def _optimize_stacking_config(X_train, y_train, X_test, y_true, opts):
    """Optimize stacking configuration with speed-focused settings"""
    global _consecutive_failures
    start_time = time.time()
    _consecutive_failures = 0

    max_acc = -np.inf
    best_f1 = 0.0
    n_samples = X_train.shape[0]

    # Smart CV fold selection based on dataset size
    if n_samples < 1000:
        cv_options = [3]  # Fast for small datasets
    elif n_samples < 5000:
        cv_options = [3, 5]  # Balanced
    else:
        cv_options = [3, 5]  # Don't go higher for speed

    # Simplified stacking configurations
    configs = []
    for cv in cv_options:
        configs.extend(
            [
                {"cv": cv, "stack_method": "auto", "passthrough": False},
                {"cv": cv, "stack_method": "predict_proba", "passthrough": False},
                {"cv": cv, "stack_method": "auto", "passthrough": True},
            ]
        )

    best_config = configs[0]

    with tqdm(configs, desc="Optimizing Stacking Config", leave=False) as pbar:
        for config in pbar:
            # Early stopping conditions
            if _consecutive_failures >= _max_consecutive_failures:
                pbar.set_description("Stacking Config (POOR ACCURACY)")
                break

            test_stacking_params = opts["stacking"].copy()
            test_stacking_params.update(config)

            accuracy, f1, success = _safe_evaluate_model(
                X_train,
                y_train,
                X_test,
                y_true,
                opts["base_estimators"],
                opts["meta_estimator"],
                **test_stacking_params,
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_config = config

            pbar.set_postfix(
                {
                    "cv": config["cv"],
                    "method": config["stack_method"][:8],
                    "pass": "Y" if config["passthrough"] else "N",
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts["stacking"].update(best_config)
    return opts, max_acc, best_f1


def _optimize_ensemble_sizes(X_train, y_train, X_test, y_true, opts):
    """Optimize ensemble sizes with much smaller, faster options"""
    global _consecutive_failures
    start_time = time.time()
    _consecutive_failures = 0

    max_acc = -np.inf
    best_f1 = 0.0
    n_samples = X_train.shape[0]

    # Much smaller ensemble sizes for stacking speed
    if n_samples < 2000:
        size_configs = [
            {"rf_n_est": 15, "et_n_est": 15, "xgb_n_est": 15},
            {"rf_n_est": 25, "et_n_est": 25, "xgb_n_est": 25},
            {"rf_n_est": 50, "et_n_est": 25, "xgb_n_est": 25},
        ]
    elif n_samples < 10000:
        size_configs = [
            {"rf_n_est": 25, "et_n_est": 25, "xgb_n_est": 25},
            {"rf_n_est": 50, "et_n_est": 50, "xgb_n_est": 50},
            {"rf_n_est": 75, "et_n_est": 50, "xgb_n_est": 50},
        ]
    else:
        size_configs = [
            {"rf_n_est": 50, "et_n_est": 50, "xgb_n_est": 50},
            {"rf_n_est": 75, "et_n_est": 75, "xgb_n_est": 75},
            {"rf_n_est": 100, "et_n_est": 75, "xgb_n_est": 75},
        ]

    best_config = size_configs[0]

    with tqdm(size_configs, desc="Optimizing Ensemble Sizes", leave=False) as pbar:
        for config in pbar:
            # Early stopping conditions
            if _consecutive_failures >= _max_consecutive_failures:
                pbar.set_description("Ensemble Sizes (POOR ACCURACY)")
                break

            # Update estimators with new sizes
            updated_estimators = []
            for name, estimator in opts["base_estimators"]:
                if name == "rf" and hasattr(estimator, "n_estimators"):
                    new_estimator = RandomForestClassifier(
                        n_estimators=config["rf_n_est"], random_state=42, n_jobs=-1
                    )
                elif name == "et" and hasattr(estimator, "n_estimators"):
                    new_estimator = ExtraTreesClassifier(
                        n_estimators=config["et_n_est"], random_state=42, n_jobs=-1
                    )
                elif (
                    name == "xgb"
                    and hasattr(estimator, "n_estimators")
                    and XGBOOST_AVAILABLE
                ):
                    new_estimator = XGBClassifier(
                        n_estimators=config["xgb_n_est"],
                        random_state=42,
                        verbosity=0,
                        n_jobs=-1,
                    )
                else:
                    new_estimator = estimator
                updated_estimators.append((name, new_estimator))

            accuracy, f1, success = _safe_evaluate_model(
                X_train,
                y_train,
                X_test,
                y_true,
                updated_estimators,
                opts["meta_estimator"],
                **opts["stacking"],
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_config = config
                opts["base_estimators"] = updated_estimators

            pbar.set_postfix(
                {
                    "rf": config["rf_n_est"],
                    "et": config.get("et_n_est", 0),
                    "xgb": config.get("xgb_n_est", 0),
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    return opts, max_acc, best_f1


def _optimize_stacking(X_train, y_train, X_test, y_true, cycles=2):
    """
    FAST optimized hyperparameters for StackingClassifier focused on your top performers.
    :param X_train: Training data
    :param y_train: Training labels
    :param X_test: Test data
    :param y_true: True labels for the test data
    :param cycles: Number of optimization cycles
    """

    n_samples, n_features = X_train.shape
    print(f"Dataset: {n_samples} samples, {n_features} features")
    print("🚨 Note: Stacking is inherently slower due to cross-validation overhead")

    # Quick baseline check with individual models
    print("Running baseline checks...")

    # Test your top performers individually
    if XGBOOST_AVAILABLE:
        baseline_xgb = XGBClassifier(n_estimators=25, random_state=42, verbosity=0)
        baseline_xgb.fit(X_train, y_train)
        baseline_acc_xgb = baseline_xgb.score(X_test, y_true)
        print(f"Baseline XGBoost: {baseline_acc_xgb:.4f}")
        best_baseline = baseline_acc_xgb
    else:
        best_baseline = 0.0

    baseline_rf = RandomForestClassifier(n_estimators=25, random_state=42)
    baseline_rf.fit(X_train, y_train)
    baseline_acc_rf = baseline_rf.score(X_test, y_true)
    print(f"Baseline RandomForest: {baseline_acc_rf:.4f}")
    best_baseline = max(best_baseline, baseline_acc_rf)

    baseline_lr = LogisticRegression(random_state=42, max_iter=500)
    baseline_lr.fit(X_train, y_train)
    baseline_acc_lr = baseline_lr.score(X_test, y_true)
    print(f"Baseline LogisticRegression: {baseline_acc_lr:.4f}")
    best_baseline = max(best_baseline, baseline_acc_lr)

    if best_baseline < 0.1:
        print("⚠️  WARNING: Very low baseline accuracy!")
        print(
            "Stacking might not help much. Consider data preprocessing or feature engineering."
        )

    # Initialize with fast defaults
    opts = {
        "stacking": {
            "cv": 3,  # Start with fast CV
            "stack_method": "auto",
            "n_jobs": -1,
            "passthrough": False,
            "verbose": 0,
        },
        "base_estimators": [],  # Will be set during optimization
        "meta_estimator": LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1),
        "meta_name": "LogisticRegression",
    }

    # Track results
    ma_vec = []
    f1_vec = []

    # Main optimization loop
    with tqdm(
        range(cycles), desc="FAST Stacking Optimization", position=0
    ) as cycle_pbar:
        for c in cycle_pbar:
            cycle_start_time = time.time()
            cycle_pbar.set_description(f"Stacking Cycle {c + 1}/{cycles}")

            # Core optimizations focused on speed
            opts, _, _ = _optimize_base_estimators(
                X_train, y_train, X_test, y_true, opts
            )
            opts, _, _ = _optimize_meta_estimator(
                X_train, y_train, X_test, y_true, opts
            )
            opts, _, _ = _optimize_stacking_config(
                X_train, y_train, X_test, y_true, opts
            )
            opts, ma, f1 = _optimize_ensemble_sizes(
                X_train, y_train, X_test, y_true, opts
            )

            ma_vec.append(ma)
            f1_vec.append(f1)

            cycle_time = time.time() - cycle_start_time

            # Display progress
            base_names = [name for name, _ in opts["base_estimators"]]
            cycle_pbar.set_postfix(
                {
                    "accuracy": f"{ma:.4f}",
                    "f1": f"{f1:.4f}",
                    "best_overall": f"{max(ma_vec):.4f}",
                    "cycle_time": f"{cycle_time:.1f}s",
                    "base_combo": "+".join(base_names)[:15],
                    "meta": opts.get("meta_name", "Unknown")[:8],
                    "cv": opts["stacking"]["cv"],
                    "method": opts["stacking"]["stack_method"][:6],
                    "passthrough": "Y" if opts["stacking"]["passthrough"] else "N",
                    "vs_baseline": f"{ma - best_baseline:+.4f}",
                }
            )

    return opts, ma, f1, ma_vec, f1_vec
