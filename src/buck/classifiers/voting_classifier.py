from typing import Any
import warnings
import numpy as np
import time
from tqdm.auto import tqdm
from sklearn.ensemble import (
    VotingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
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

# Global efficiency controls
_max_time_per_step = 300  # 5 minutes max per optimization step
_min_accuracy_threshold = 0.25  # Stop if accuracy is consistently terrible
_consecutive_failures = 0
_max_consecutive_failures = 3


def _safe_evaluate_model(X_train, y_train, X_test, y_true, estimators, **voting_params):
    """Safely evaluate a voting model configuration with timeout protection"""
    global _consecutive_failures

    try:
        start_time = time.time()
        classifier = VotingClassifier(estimators=estimators, **voting_params)
        classifier.fit(X_train, y_train)

        # Check if training took too long
        if time.time() - start_time > 120:  # 2 minutes max per model
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
        print(f"Model evaluation failed: {e}")
        _consecutive_failures += 1
        return 0.0, 0.0, False


def _create_top_performer_combinations(n_samples, n_features):
    """Create combinations focused on YOUR TOP 3 PERFORMERS: XGBoost, RF, ET"""

    # Your proven top performers with optimized settings
    if XGBOOST_AVAILABLE:
        if n_samples < 5000:
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
                        n_estimators=100, random_state=42, verbosity=0, n_jobs=-1
                    ),
                ),
            ]
    else:
        xgb_estimators = []

    # Your top tree performers - optimized sizes
    if n_samples < 5000:
        tree_estimators = [
            ("rf", RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)),
            ("et", ExtraTreesClassifier(n_estimators=50, random_state=42, n_jobs=-1)),
        ]
    else:
        tree_estimators = [
            (
                "rf",
                RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            ),
            ("et", ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ]

    # Fast supporting algorithms (only as supplements)
    fast_estimators = [
        ("lr", LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1)),
        ("nb", GaussianNB()),
    ]

    # FOCUS ON YOUR TOP PERFORMERS - build combinations around them
    combinations = []

    if XGBOOST_AVAILABLE:
        # Your top 3 together (the dream team!)
        combinations.append(xgb_estimators + tree_estimators)  # xgb + rf + et

        # Top performer pairs
        combinations.append(xgb_estimators + tree_estimators[:1])  # xgb + rf
        combinations.append(xgb_estimators + tree_estimators[1:])  # xgb + et

        # Add one fast model to top performers
        combinations.append(
            xgb_estimators + tree_estimators[:1] + fast_estimators[:1]
        )  # xgb + rf + lr
        combinations.append(
            xgb_estimators + tree_estimators + fast_estimators[:1]
        )  # all 4

    # Your proven tree performers
    combinations.append(tree_estimators)  # rf + et
    combinations.append(tree_estimators[:1] + fast_estimators[:1])  # rf + lr
    combinations.append(tree_estimators[1:] + fast_estimators[:1])  # et + lr

    # Fallback: fast models only (if top performers fail)
    combinations.append(fast_estimators)  # lr + nb

    return combinations


def _optimize_estimator_combinations(X_train, y_train, X_test, y_true, opts):
    """Optimize combinations focused on your top performers"""
    global _consecutive_failures
    start_time = time.time()
    _consecutive_failures = 0

    max_acc = -np.inf
    best_f1 = 0.0
    n_samples, n_features = X_train.shape

    # Get combinations focused on your top performers
    estimator_combinations = _create_top_performer_combinations(n_samples, n_features)
    best_estimators = estimator_combinations[0]

    with tqdm(
        estimator_combinations,
        desc="Optimizing TOP PERFORMER Combinations",
        leave=False,
    ) as pbar:
        for estimators in pbar:
            # Early stopping conditions
            if time.time() - start_time > _max_time_per_step:
                pbar.set_description("Top Performers (TIME LIMIT)")
                break
            if _consecutive_failures >= _max_consecutive_failures:
                pbar.set_description("Top Performers (POOR ACCURACY)")
                break

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, estimators, **opts["voting"]
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

    opts["estimators"] = best_estimators
    return opts, max_acc, best_f1


def _optimize_voting_config(X_train, y_train, X_test, y_true, opts):
    """Optimize voting configuration"""
    global _consecutive_failures
    start_time = time.time()
    _consecutive_failures = 0

    max_acc = -np.inf
    best_f1 = 0.0

    # Test if all estimators support predict_proba
    all_support_proba = True
    for name, estimator in opts["estimators"]:
        if not hasattr(estimator, "predict_proba"):
            all_support_proba = False
            break

    # Voting configurations
    if all_support_proba:
        configs = [
            {"voting": "soft", "flatten_transform": True},
            {"voting": "hard", "flatten_transform": True},
            {
                "voting": "soft",
                "flatten_transform": False,
            },  # Sometimes better for XGBoost
        ]
    else:
        configs = [
            {"voting": "hard", "flatten_transform": True},
        ]

    best_config = configs[0]

    with tqdm(configs, desc="Optimizing Voting Strategy", leave=False) as pbar:
        for config in pbar:
            # Early stopping conditions
            if time.time() - start_time > _max_time_per_step:
                pbar.set_description("Voting Strategy (TIME LIMIT)")
                break
            if _consecutive_failures >= _max_consecutive_failures:
                pbar.set_description("Voting Strategy (POOR ACCURACY)")
                break

            test_voting_params = opts["voting"].copy()
            test_voting_params.update(config)

            accuracy, f1, success = _safe_evaluate_model(
                X_train,
                y_train,
                X_test,
                y_true,
                opts["estimators"],
                **test_voting_params,
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_config = config

            pbar.set_postfix(
                {
                    "voting": config["voting"],
                    "flatten": "Y" if config["flatten_transform"] else "N",
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts["voting"].update(best_config)
    return opts, max_acc, best_f1


def _optimize_top_performer_params(X_train, y_train, X_test, y_true, opts):
    """Optimize parameters for your top performers: XGBoost, RF, ET"""
    global _consecutive_failures
    start_time = time.time()
    _consecutive_failures = 0

    max_acc = -np.inf
    best_f1 = 0.0
    n_samples = X_train.shape[0]

    # Parameter configs optimized for your top performers
    if n_samples < 2000:
        param_configs = [
            # Small dataset configs
            {"n_estimators": 25, "max_depth": 6, "learning_rate": 0.1},
            {"n_estimators": 50, "max_depth": 8, "learning_rate": 0.1},
            {"n_estimators": 75, "max_depth": 10, "learning_rate": 0.05},
        ]
    elif n_samples < 10000:
        param_configs = [
            # Medium dataset configs
            {"n_estimators": 50, "max_depth": 8, "learning_rate": 0.1},
            {"n_estimators": 100, "max_depth": 10, "learning_rate": 0.1},
            {"n_estimators": 150, "max_depth": 12, "learning_rate": 0.05},
        ]
    else:
        param_configs = [
            # Large dataset configs
            {"n_estimators": 100, "max_depth": 10, "learning_rate": 0.1},
            {"n_estimators": 200, "max_depth": 12, "learning_rate": 0.05},
            {"n_estimators": 300, "max_depth": 15, "learning_rate": 0.05},
        ]

    best_config = param_configs[0]

    with tqdm(
        param_configs, desc="Optimizing Top Performer Params", leave=False
    ) as pbar:
        for config in pbar:
            # Early stopping conditions
            if time.time() - start_time > _max_time_per_step:
                pbar.set_description("Top Performer Params (TIME LIMIT)")
                break
            if _consecutive_failures >= _max_consecutive_failures:
                pbar.set_description("Top Performer Params (POOR ACCURACY)")
                break

            # Update estimators with new parameters
            updated_estimators = []
            for name, estimator in opts["estimators"]:
                if name == "xgb" and XGBOOST_AVAILABLE:
                    new_estimator = XGBClassifier(
                        n_estimators=config["n_estimators"],
                        max_depth=config["max_depth"],
                        learning_rate=config["learning_rate"],
                        random_state=42,
                        verbosity=0,
                        n_jobs=-1,
                    )
                elif name == "rf":
                    new_estimator = RandomForestClassifier(
                        n_estimators=config["n_estimators"],
                        max_depth=config["max_depth"],
                        random_state=42,
                        n_jobs=-1,
                    )
                elif name == "et":
                    new_estimator = ExtraTreesClassifier(
                        n_estimators=config["n_estimators"],
                        max_depth=config["max_depth"],
                        random_state=42,
                        n_jobs=-1,
                    )
                else:
                    new_estimator = estimator  # Keep other estimators unchanged

                updated_estimators.append((name, new_estimator))

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, updated_estimators, **opts["voting"]
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_config = config
                opts["estimators"] = updated_estimators

            pbar.set_postfix(
                {
                    "n_est": config["n_estimators"],
                    "depth": config["max_depth"],
                    "lr": f"{config['learning_rate']:.2f}",
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    return opts, max_acc, best_f1


def _optimize_voting(X_train, y_train, X_test, y_true, cycles=2):
    """
    Optimizes VotingClassifier focused on YOUR TOP PERFORMERS: XGBoost, RF, ET
    :param X_train: Training data
    :param y_train: Training labels
    :param X_test: Test data
    :param y_true: True labels for the test data
    :param cycles: Number of optimization cycles
    """

    # Quick baseline check with your top performer
    print("Running baseline check with your top performers...")

    if XGBOOST_AVAILABLE:
        baseline_xgb = XGBClassifier(n_estimators=25, random_state=42, verbosity=0)
        baseline_xgb.fit(X_train, y_train)
        baseline_acc_xgb = baseline_xgb.score(X_test, y_true)
        print(f"Baseline XGBoost accuracy: {baseline_acc_xgb:.4f}")

    baseline_rf = RandomForestClassifier(n_estimators=25, random_state=42)
    baseline_rf.fit(X_train, y_train)
    baseline_acc_rf = baseline_rf.score(X_test, y_true)
    print(f"Baseline RandomForest accuracy: {baseline_acc_rf:.4f}")

    baseline_et = ExtraTreesClassifier(n_estimators=25, random_state=42)
    baseline_et.fit(X_train, y_train)
    baseline_acc_et = baseline_et.score(X_test, y_true)
    print(f"Baseline ExtraTrees accuracy: {baseline_acc_et:.4f}")

    best_baseline = max(baseline_acc_rf, baseline_acc_et)
    if XGBOOST_AVAILABLE:
        best_baseline = max(best_baseline, baseline_acc_xgb)

    if best_baseline < 0.2:
        print("⚠️  WARNING: Very low baseline accuracy even with your top performers!")
        print(
            "Consider checking data preprocessing, feature engineering, or class balance."
        )

    # Initialize with optimized defaults for your top performers
    opts = {
        "voting": {
            "voting": "soft",  # Usually better for XGBoost + RF + ET
            "n_jobs": -1,
            "flatten_transform": True,
            "verbose": 0,
        },
        "estimators": [],  # Will be set during optimization
    }

    # Track results
    ma_vec = []
    f1_vec = []

    # Main optimization loop
    with tqdm(
        range(cycles), desc="TOP PERFORMER Voting Optimization", position=0
    ) as cycle_pbar:
        for c in cycle_pbar:
            cycle_start_time = time.time()
            cycle_pbar.set_description(f"Top Performer Cycle {c + 1}/{cycles}")

            # Optimizations focused on your proven winners
            opts, _, _ = _optimize_estimator_combinations(
                X_train, y_train, X_test, y_true, opts
            )
            opts, _, _ = _optimize_voting_config(X_train, y_train, X_test, y_true, opts)
            opts, ma, f1 = _optimize_top_performer_params(
                X_train, y_train, X_test, y_true, opts
            )

            ma_vec.append(ma)
            f1_vec.append(f1)

            cycle_time = time.time() - cycle_start_time

            # Display progress
            estimator_names = [name for name, _ in opts["estimators"]]
            cycle_pbar.set_postfix(
                {
                    "accuracy": f"{ma:.4f}",
                    "f1": f"{f1:.4f}",
                    "best_overall": f"{max(ma_vec):.4f}",
                    "cycle_time": f"{cycle_time:.1f}s",
                    "combo": "+".join(estimator_names)[:15],
                    "voting": opts["voting"]["voting"],
                    "baseline_beat": f"{ma - best_baseline:+.4f}",
                }
            )

    return opts, ma, f1, ma_vec, f1_vec


def _analyze_voting_performance(X_train, y_train, X_test, y_true, best_opts):
    """Analyze the performance of voting vs individual top performers"""

    print("\n" + "=" * 70)
    print("TOP PERFORMER VOTING CLASSIFIER ANALYSIS")
    print("=" * 70)

    # Train voting classifier
    voting_clf = VotingClassifier(
        estimators=best_opts["estimators"], **best_opts["voting"]
    )
    voting_clf.fit(X_train, y_train)
    y_pred_voting = voting_clf.predict(X_test)

    acc_voting = accuracy_score(y_true, y_pred_voting)
    f1_voting = f1_score(y_true, y_pred_voting, average="weighted")

    # Train individual estimators
    print(f"Individual Top Performer Results:")
    individual_performances = []

    for name, estimator in best_opts["estimators"]:
        estimator.fit(X_train, y_train)
        y_pred_individual = estimator.predict(X_test)
        acc_individual = accuracy_score(y_true, y_pred_individual)
        f1_individual = f1_score(y_true, y_pred_individual, average="weighted")
        individual_performances.append((name, acc_individual, f1_individual))

        # Special highlighting for your top 3
        marker = "🏆" if name in ["xgb", "rf", "et"] else "  "
        print(
            f"  {marker} {name:12s}: Accuracy={acc_individual:.4f}, F1={f1_individual:.4f}"
        )

    # Voting performance
    print(f"\n🎯 Optimized Voting Classifier:")
    print(f"  Voting Method: {best_opts['voting']['voting']}")
    print(f"  Accuracy: {acc_voting:.4f}")
    print(f"  F1 Score: {f1_voting:.4f}")

    # Best individual estimator
    best_individual = max(individual_performances, key=lambda x: x[1])
    best_name, best_acc, best_f1 = best_individual

    print(f"\n🥇 Best Individual: {best_name}")
    print(f"  Accuracy: {best_acc:.4f}")
    print(f"  F1 Score: {best_f1:.4f}")

    # Improvement analysis
    acc_improvement = acc_voting - best_acc
    f1_improvement = f1_voting - best_f1

    print(f"\n📈 Voting Improvement:")
    print(
        f"  Accuracy: {acc_improvement:+.4f} ({acc_improvement / best_acc * 100:+.1f}%)"
    )
    print(f"  F1 Score: {f1_improvement:+.4f} ({f1_improvement / best_f1 * 100:+.1f}%)")

    # Check if ensemble beats all individuals
    all_individual_accs = [perf[1] for perf in individual_performances]
    if acc_voting > max(all_individual_accs):
        print("  ✅ Voting beats ALL individual models!")
    else:
        print("  ⚠️  Some individual models still outperform voting")

    return {
        "voting_accuracy": acc_voting,
        "voting_f1": f1_voting,
        "best_individual_accuracy": best_acc,
        "best_individual_f1": best_f1,
        "accuracy_improvement": acc_improvement,
        "f1_improvement": f1_improvement,
        "individual_performances": individual_performances,
    }


# Example usage function
def example_usage():
    """Example using the TOP PERFORMER focused optimization"""
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    print("🚀 TOP PERFORMER Voting Classifier Optimization")
    print("Focusing on: XGBoost, RandomForest, ExtraTrees")
    print("=" * 50)

    # Generate sample data
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        random_state=42,
    )

    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(
        f"Dataset: {X_train_scaled.shape[0]} samples, {X_train_scaled.shape[1]} features"
    )

    # Run optimization focused on top performers
    best_opts, best_acc, best_f1, acc_history, f1_history = _optimize_voting(
        X_train_scaled, y_train, X_test_scaled, y_test, cycles=1
    )

    print(f"\n🎯 FINAL RESULTS:")
    print(f"Best Voting Accuracy: {best_acc:.4f}")
    print(f"Best Voting F1: {best_f1:.4f}")

    # Detailed analysis
    analysis = _analyze_voting_performance(
        X_train_scaled, y_train, X_test_scaled, y_test, best_opts
    )

    return best_opts, best_acc, best_f1, acc_history, f1_history
