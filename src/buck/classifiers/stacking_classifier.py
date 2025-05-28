from typing import Any
import warnings
import numpy as np
from tqdm.auto import tqdm
from sklearn.ensemble import (
    StackingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.exceptions import ConvergenceWarning

# Suppress convergence warnings for cleaner progress bars
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def _safe_evaluate_model(
    X_train, y_train, X_test, y_true, estimators, final_estimator, **stacking_params
):
    """Safely evaluate a stacking model configuration"""
    try:
        classifier = StackingClassifier(
            estimators=estimators, final_estimator=final_estimator, **stacking_params
        )
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        return accuracy, f1, True
    except Exception:
        return 0.0, 0.0, False


def _create_base_estimator_combinations(n_samples, n_features):
    """Create different base estimator combinations based on data characteristics"""

    # Fast estimators for large datasets
    fast_estimators = [
        ("lr", LogisticRegression(random_state=42, max_iter=1000)),
        ("nb", GaussianNB()),
        ("sgd", SGDClassifier(random_state=42, max_iter=1000)),
    ]

    # Tree-based estimators
    tree_estimators = [
        ("rf", RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)),
        ("et", ExtraTreesClassifier(n_estimators=50, random_state=42, n_jobs=-1)),
        ("dt", DecisionTreeClassifier(random_state=42)),
    ]

    # Advanced estimators
    advanced_estimators = [
        ("gb", GradientBoostingClassifier(n_estimators=50, random_state=42)),
        ("svc", SVC(probability=True, random_state=42)),
    ]

    # Additional estimators for smaller datasets
    if n_samples < 10000:
        additional_estimators = [
            ("knn", KNeighborsClassifier(n_neighbors=min(5, max(3, n_samples // 100)))),
        ]
    else:
        additional_estimators = []

    # Different combinations based on dataset size
    if n_samples > 50000:  # Large dataset: use fast estimators
        combinations = [
            fast_estimators[:2],  # lr, nb
            fast_estimators[:3],  # lr, nb, sgd
            fast_estimators + tree_estimators[:1],  # lr, nb, sgd, rf
        ]
    elif n_samples > 10000:  # Medium dataset: balanced approach
        combinations = [
            fast_estimators[:2] + tree_estimators[:2],  # lr, nb, rf, et
            fast_estimators[:3] + tree_estimators[:1],  # lr, nb, sgd, rf
            tree_estimators[:3],  # rf, et, dt
            fast_estimators[:2] + advanced_estimators[:1],  # lr, nb, gb
        ]
    else:  # Small dataset: can use all estimators
        combinations = [
            fast_estimators[:3],  # lr, nb, sgd
            tree_estimators[:3],  # rf, et, dt
            fast_estimators[:2] + tree_estimators[:2],  # lr, nb, rf, et
            fast_estimators[:2] + advanced_estimators,  # lr, nb, gb, svc
            tree_estimators[:2] + advanced_estimators[:1],  # rf, et, gb
        ]

        if additional_estimators:
            combinations.append(
                fast_estimators[:2] + additional_estimators
            )  # lr, nb, knn

    return combinations


def _optimize_base_estimators(X_train, y_train, X_test, y_true, opts):
    """Optimize base estimator combinations"""
    max_acc = -np.inf
    best_f1 = 0.0
    n_samples, n_features = X_train.shape

    # Get estimator combinations
    estimator_combinations = _create_base_estimator_combinations(n_samples, n_features)
    best_estimators = estimator_combinations[0]

    with tqdm(
        estimator_combinations, desc="Optimizing Base Estimators", leave=False
    ) as pbar:
        for estimators in pbar:
            # Create final estimator
            final_estimator = LogisticRegression(**opts["meta"])

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
                    "estimators": "+".join(estimator_names)[:15],
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts["base_estimators"] = best_estimators
    return opts, max_acc, best_f1


def _optimize_meta_estimator(X_train, y_train, X_test, y_true, opts):
    """Optimize meta estimator (final estimator)"""
    max_acc = -np.inf
    best_f1 = 0.0

    # Different meta estimators
    meta_estimators = [
        ("LogisticRegression", LogisticRegression(random_state=42, max_iter=1000)),
        (
            "LogisticRegression_L1",
            LogisticRegression(
                penalty="l1", solver="liblinear", random_state=42, max_iter=1000
            ),
        ),
        (
            "RandomForest",
            RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
        ),
        (
            "GradientBoosting",
            GradientBoostingClassifier(n_estimators=50, random_state=42),
        ),
        ("SGD", SGDClassifier(random_state=42, max_iter=1000)),
    ]

    best_meta = meta_estimators[0]

    with tqdm(meta_estimators, desc="Optimizing Meta Estimator", leave=False) as pbar:
        for name, meta_estimator in pbar:
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
    """Optimize stacking configuration parameters together"""
    max_acc = -np.inf
    best_f1 = 0.0

    # Combined stacking configurations
    configs = [
        {"cv": 3, "stack_method": "auto", "passthrough": False},
        {"cv": 5, "stack_method": "auto", "passthrough": False},
        {"cv": 5, "stack_method": "predict_proba", "passthrough": False},
        {"cv": 5, "stack_method": "auto", "passthrough": True},
        {"cv": 3, "stack_method": "predict_proba", "passthrough": True},
        {"cv": 7, "stack_method": "auto", "passthrough": False},
        {"cv": 5, "stack_method": "predict", "passthrough": False},
    ]
    best_config = configs[0]

    with tqdm(configs, desc="Optimizing Stacking Config", leave=False) as pbar:
        for config in pbar:
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
    """Optimize ensemble sizes for tree-based estimators"""
    max_acc = -np.inf
    best_f1 = 0.0

    # Different ensemble size configurations
    size_configs = [
        {"rf_n_est": 50, "et_n_est": 50, "gb_n_est": 50},
        {"rf_n_est": 100, "et_n_est": 100, "gb_n_est": 50},
        {"rf_n_est": 200, "et_n_est": 100, "gb_n_est": 100},
        {"rf_n_est": 100, "et_n_est": 200, "gb_n_est": 50},
        {"rf_n_est": 150, "et_n_est": 150, "gb_n_est": 75},
    ]
    best_config = size_configs[0]

    with tqdm(size_configs, desc="Optimizing Ensemble Sizes", leave=False) as pbar:
        for config in pbar:
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
                elif name == "gb" and hasattr(estimator, "n_estimators"):
                    new_estimator = GradientBoostingClassifier(
                        n_estimators=config["gb_n_est"], random_state=42
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
                    "et": config["et_n_est"],
                    "gb": config["gb_n_est"],
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    return opts, max_acc, best_f1


def _optimize_regularization(X_train, y_train, X_test, y_true, opts):
    """Optimize regularization for meta estimator"""
    max_acc = -np.inf
    best_f1 = 0.0

    # Only optimize if meta estimator is LogisticRegression
    if not isinstance(opts["meta_estimator"], LogisticRegression):
        return opts, max_acc, best_f1

    # Different C values for LogisticRegression
    C_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    best_C = C_values[0]

    with tqdm(C_values, desc="Optimizing Meta Regularization", leave=False) as pbar:
        for C in pbar:
            meta_estimator = LogisticRegression(C=C, random_state=42, max_iter=1000)

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
                best_C = C
                opts["meta_estimator"] = meta_estimator

            pbar.set_postfix(
                {
                    "C": f"{C:.1f}",
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    return opts, max_acc, best_f1


def _optimize_stacking(X_train, y_train, X_test, y_true, cycles=2):
    """
    Optimizes the hyperparameters for a StackingClassifier.
    :param X_train: Training data
    :param y_train: Training labels
    :param X_test: Test data
    :param y_true: True labels for the test data
    :param cycles: Number of optimization cycles
    """

    # Initialize default parameters
    opts = {
        "stacking": {
            "cv": 5,
            "stack_method": "auto",
            "n_jobs": -1,
            "passthrough": False,
            "verbose": 0,
        },
        "meta": {
            "C": 1.0,
            "max_iter": 1000,
            "random_state": 42,
        },
        "base_estimators": [],  # Will be set during optimization
        "meta_estimator": LogisticRegression(random_state=42, max_iter=1000),
        "meta_name": "LogisticRegression",
    }

    # Track results
    ma_vec = []
    f1_vec = []

    # Main optimization loop with overall progress bar
    with tqdm(
        range(cycles), desc="Stacking Classifier Optimization Cycles", position=0
    ) as cycle_pbar:
        for c in cycle_pbar:
            cycle_pbar.set_description(f"Stacking Cycle {c + 1}/{cycles}")

            # Core optimizations
            opts, _, _ = _optimize_base_estimators(
                X_train, y_train, X_test, y_true, opts
            )
            opts, _, _ = _optimize_meta_estimator(
                X_train, y_train, X_test, y_true, opts
            )
            opts, _, _ = _optimize_stacking_config(
                X_train, y_train, X_test, y_true, opts
            )

            # Fine-tuning optimizations
            opts, _, _ = _optimize_ensemble_sizes(
                X_train, y_train, X_test, y_true, opts
            )
            opts, ma, f1 = _optimize_regularization(
                X_train, y_train, X_test, y_true, opts
            )

            ma_vec.append(ma)
            f1_vec.append(f1)

            # Display progress
            base_names = [name for name, _ in opts["base_estimators"]]
            cycle_pbar.set_postfix(
                {
                    "accuracy": f"{ma:.4f}",
                    "f1": f"{f1:.4f}",
                    "best_overall": f"{max(ma_vec):.4f}",
                    "base_est": "+".join(base_names)[:15],
                    "meta": opts.get("meta_name", "Unknown")[:8],
                    "cv": opts["stacking"]["cv"],
                    "method": opts["stacking"]["stack_method"][:6],
                    "passthrough": "Y" if opts["stacking"]["passthrough"] else "N",
                }
            )

    return opts, ma, f1, ma_vec, f1_vec


def _analyze_stacking_performance(X_train, y_train, X_test, y_true, best_opts):
    """Analyze the performance of stacking vs individual base estimators"""

    print("\n" + "=" * 70)
    print("STACKING CLASSIFIER PERFORMANCE ANALYSIS")
    print("=" * 70)

    # Train stacking classifier
    stacking_clf = StackingClassifier(
        estimators=best_opts["base_estimators"],
        final_estimator=best_opts["meta_estimator"],
        **best_opts["stacking"],
    )
    stacking_clf.fit(X_train, y_train)
    y_pred_stacking = stacking_clf.predict(X_test)

    acc_stacking = accuracy_score(y_true, y_pred_stacking)
    f1_stacking = f1_score(y_true, y_pred_stacking, average="weighted")

    # Train individual base estimators
    print(f"Base Estimators Performance:")
    base_performances = []

    for name, estimator in best_opts["base_estimators"]:
        estimator.fit(X_train, y_train)
        y_pred_base = estimator.predict(X_test)
        acc_base = accuracy_score(y_true, y_pred_base)
        f1_base = f1_score(y_true, y_pred_base, average="weighted")
        base_performances.append((name, acc_base, f1_base))
        print(f"  {name:15s}: Accuracy={acc_base:.4f}, F1={f1_base:.4f}")

    # Meta estimator trained on true labels (not cross-validated predictions)
    print(
        f"\nMeta Estimator ({best_opts.get('meta_name', 'Unknown')}) on original data:"
    )
    meta_direct = best_opts["meta_estimator"]
    meta_direct.fit(X_train, y_train)
    y_pred_meta = meta_direct.predict(X_test)
    acc_meta = accuracy_score(y_true, y_pred_meta)
    f1_meta = f1_score(y_true, y_pred_meta, average="weighted")
    print(f"  Direct Training: Accuracy={acc_meta:.4f}, F1={f1_meta:.4f}")

    # Stacking performance
    print(f"\nStacking Classifier:")
    print(f"  Accuracy: {acc_stacking:.4f}")
    print(f"  F1 Score: {f1_stacking:.4f}")

    # Best individual estimator
    best_individual = max(base_performances, key=lambda x: x[1])
    best_name, best_acc, best_f1 = best_individual

    print(f"\nBest Individual Estimator: {best_name}")
    print(f"  Accuracy: {best_acc:.4f}")
    print(f"  F1 Score: {best_f1:.4f}")

    # Improvement analysis
    acc_improvement = acc_stacking - best_acc
    f1_improvement = f1_stacking - best_f1

    print(f"\nStacking Improvement over Best Individual:")
    print(
        f"  Accuracy: {acc_improvement:+.4f} ({acc_improvement / best_acc * 100:+.1f}%)"
    )
    print(f"  F1 Score: {f1_improvement:+.4f} ({f1_improvement / best_f1 * 100:+.1f}%)")

    # Configuration summary
    print(f"\nStacking Configuration:")
    print(f"  Base Estimators: {len(best_opts['base_estimators'])}")
    print(f"  CV Folds: {best_opts['stacking']['cv']}")
    print(f"  Stack Method: {best_opts['stacking']['stack_method']}")
    print(f"  Passthrough: {best_opts['stacking']['passthrough']}")
    print(f"  Meta Estimator: {best_opts.get('meta_name', 'Unknown')}")

    return {
        "stacking_accuracy": acc_stacking,
        "stacking_f1": f1_stacking,
        "best_individual_accuracy": best_acc,
        "best_individual_f1": best_f1,
        "accuracy_improvement": acc_improvement,
        "f1_improvement": f1_improvement,
        "base_performances": base_performances,
    }


# Example usage function
def example_usage():
    """Example of how to use the optimized Stacking Classifier function"""
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # Generate sample data
    print("Generating sample classification data...")
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        random_state=42,
    )

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale the data (important for some estimators)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Training set: {X_train_scaled.shape}")
    print(f"Test set: {X_test_scaled.shape}")
    print("Starting Stacking Classifier optimization...")

    # Run optimization
    best_opts, best_acc, best_f1, acc_history, f1_history = (
        _optimize_stacking_classifier(
            X_train_scaled, y_train, X_test_scaled, y_test, cycles=2
        )
    )

    print(f"\nOptimization completed!")
    print(f"Best accuracy: {best_acc:.4f}")
    print(f"Best F1 score: {best_f1:.4f}")

    # Analyze performance
    analysis = _analyze_stacking_performance(
        X_train_scaled, y_train, X_test_scaled, y_test, best_opts
    )

    return best_opts, best_acc, best_f1, acc_history, f1_history


if __name__ == "__main__":
    # Run example
    example_usage()
