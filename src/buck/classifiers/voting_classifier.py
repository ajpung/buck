from typing import Any
import warnings
import numpy as np
from tqdm.auto import tqdm
from sklearn.ensemble import (
    VotingClassifier,
    BaggingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.exceptions import ConvergenceWarning

# Suppress convergence warnings for cleaner progress bars
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def _safe_evaluate_model(X_train, y_train, X_test, y_true, estimators, **voting_params):
    """Safely evaluate a voting model configuration"""
    try:
        classifier = VotingClassifier(estimators=estimators, **voting_params)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        return accuracy, f1, True
    except Exception:
        return 0.0, 0.0, False


def _create_estimator_combinations(n_samples, n_features):
    """Create different estimator combinations based on data characteristics"""

    # Fast estimators
    fast_estimators = [
        ("lr", LogisticRegression(random_state=42, max_iter=1000)),
        ("nb", GaussianNB()),
    ]

    # Tree-based estimators (core strength)
    tree_estimators = [
        ("rf", RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)),
        ("et", ExtraTreesClassifier(n_estimators=50, random_state=42, n_jobs=-1)),
        ("dt", DecisionTreeClassifier(random_state=42)),
    ]

    # Ensemble estimators
    ensemble_estimators = [
        ("bagging", BaggingClassifier(n_estimators=50, random_state=42, n_jobs=-1)),
        ("gb", GradientBoostingClassifier(n_estimators=50, random_state=42)),
    ]

    # Advanced estimators
    if n_samples < 10000:  # Only for smaller datasets due to computational cost
        advanced_estimators = [
            ("svc", SVC(probability=True, random_state=42)),
            ("knn", KNeighborsClassifier(n_neighbors=min(5, max(3, n_samples // 100)))),
        ]
    else:
        advanced_estimators = []

    # Different combinations based on dataset characteristics
    if n_samples > 50000:  # Large dataset: focus on fast estimators
        combinations = [
            fast_estimators + tree_estimators[:2],  # lr, nb, rf, et
            tree_estimators[:2] + ensemble_estimators[:1],  # rf, et, bagging
            fast_estimators[:1]
            + tree_estimators[:2]
            + ensemble_estimators[:1],  # lr, rf, et, bagging
        ]
    elif n_samples > 10000:  # Medium dataset: balanced approach
        combinations = [
            tree_estimators[:3],  # rf, et, dt
            fast_estimators + tree_estimators[:2],  # lr, nb, rf, et
            tree_estimators[:2] + ensemble_estimators,  # rf, et, bagging, gb
            fast_estimators[:1]
            + tree_estimators[:2]
            + ensemble_estimators[:1],  # lr, rf, et, bagging
            fast_estimators + ensemble_estimators[:1],  # lr, nb, bagging
        ]
    else:  # Small dataset: can use all estimators
        combinations = [
            tree_estimators[:3],  # rf, et, dt
            fast_estimators + tree_estimators[:2],  # lr, nb, rf, et
            tree_estimators[:2] + ensemble_estimators,  # rf, et, bagging, gb
            fast_estimators + advanced_estimators,  # lr, nb, svc, knn
            tree_estimators[:2] + advanced_estimators[:1],  # rf, et, svc
            fast_estimators[:1]
            + tree_estimators[:1]
            + ensemble_estimators[:1]
            + advanced_estimators[:1],
            # lr, rf, bagging, svc
        ]

        if advanced_estimators:
            combinations.append(
                fast_estimators[:1] + advanced_estimators
            )  # lr, svc, knn

    return combinations


def _optimize_estimator_combinations(X_train, y_train, X_test, y_true, opts):
    """Optimize estimator combinations"""
    max_acc = -np.inf
    best_f1 = 0.0
    n_samples, n_features = X_train.shape

    # Get estimator combinations
    estimator_combinations = _create_estimator_combinations(n_samples, n_features)
    best_estimators = estimator_combinations[0]

    with tqdm(
        estimator_combinations, desc="Optimizing Estimator Combinations", leave=False
    ) as pbar:
        for estimators in pbar:
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
                    "estimators": "+".join(estimator_names)[:15],
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts["estimators"] = best_estimators
    return opts, max_acc, best_f1


def _optimize_voting_config(X_train, y_train, X_test, y_true, opts):
    """Optimize voting configuration"""
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
            {"voting": "soft", "flatten_transform": False},
            {"voting": "hard", "flatten_transform": True},
        ]
    else:
        configs = [
            {"voting": "hard", "flatten_transform": True},
            {"voting": "hard", "flatten_transform": False},
        ]

    best_config = configs[0]

    with tqdm(configs, desc="Optimizing Voting Config", leave=False) as pbar:
        for config in pbar:
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


def _optimize_ensemble_sizes(X_train, y_train, X_test, y_true, opts):
    """Optimize ensemble sizes for tree-based estimators"""
    max_acc = -np.inf
    best_f1 = 0.0

    # Different ensemble size configurations
    size_configs = [
        {"default": 50},  # Small and fast
        {"default": 100},  # Balanced
        {"default": 200},  # Large and powerful
        {"rf": 100, "et": 50, "bagging": 75, "gb": 50},  # Mixed sizes
        {"rf": 150, "et": 100, "bagging": 100, "gb": 75},  # Larger mixed
    ]
    best_config = size_configs[0]

    with tqdm(size_configs, desc="Optimizing Ensemble Sizes", leave=False) as pbar:
        for config in pbar:
            # Update estimators with new sizes
            updated_estimators = []
            for name, estimator in opts["estimators"]:
                if hasattr(estimator, "n_estimators"):
                    # Get size for this estimator type
                    if name in config:
                        new_size = config[name]
                    elif "default" in config:
                        new_size = config["default"]
                    else:
                        new_size = 50  # fallback

                    # Create new estimator with updated size
                    if name == "rf":
                        new_estimator = RandomForestClassifier(
                            n_estimators=new_size, random_state=42, n_jobs=-1
                        )
                    elif name == "et":
                        new_estimator = ExtraTreesClassifier(
                            n_estimators=new_size, random_state=42, n_jobs=-1
                        )
                    elif name == "bagging":
                        new_estimator = BaggingClassifier(
                            n_estimators=new_size, random_state=42, n_jobs=-1
                        )
                    elif name == "gb":
                        new_estimator = GradientBoostingClassifier(
                            n_estimators=new_size, random_state=42
                        )
                    else:
                        new_estimator = estimator  # Keep original if unknown type
                else:
                    new_estimator = estimator  # Keep non-ensemble estimators unchanged

                updated_estimators.append((name, new_estimator))

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, updated_estimators, **opts["voting"]
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_config = config
                opts["estimators"] = updated_estimators

            if "default" in config:
                size_str = f"all={config['default']}"
            else:
                size_str = f"rf={config.get('rf', 50)},et={config.get('et', 50)}"

            pbar.set_postfix(
                {
                    "sizes": size_str[:12],
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    return opts, max_acc, best_f1


def _optimize_tree_parameters(X_train, y_train, X_test, y_true, opts):
    """Optimize tree-based parameters for ensemble estimators"""
    max_acc = -np.inf
    best_f1 = 0.0

    # Tree parameter configurations
    tree_configs = [
        {"max_depth": None, "max_features": "sqrt", "min_samples_split": 2},
        {"max_depth": 10, "max_features": "sqrt", "min_samples_split": 2},
        {"max_depth": 20, "max_features": "sqrt", "min_samples_split": 2},
        {"max_depth": None, "max_features": "log2", "min_samples_split": 2},
        {"max_depth": None, "max_features": None, "min_samples_split": 2},
        {"max_depth": None, "max_features": "sqrt", "min_samples_split": 5},
        {"max_depth": 15, "max_features": "sqrt", "min_samples_split": 5},
    ]
    best_config = tree_configs[0]

    with tqdm(tree_configs, desc="Optimizing Tree Parameters", leave=False) as pbar:
        for config in pbar:
            # Update tree-based estimators with new parameters
            updated_estimators = []
            for name, estimator in opts["estimators"]:
                if name in ["rf", "et", "dt"] and hasattr(estimator, "max_depth"):
                    # Get current n_estimators if it exists
                    n_est = getattr(estimator, "n_estimators", 50)

                    if name == "rf":
                        new_estimator = RandomForestClassifier(
                            n_estimators=n_est, random_state=42, n_jobs=-1, **config
                        )
                    elif name == "et":
                        new_estimator = ExtraTreesClassifier(
                            n_estimators=n_est, random_state=42, n_jobs=-1, **config
                        )
                    elif name == "dt":
                        tree_config = {
                            k: v for k, v in config.items() if k != "min_samples_split"
                        }  # DT doesn't always use this
                        new_estimator = DecisionTreeClassifier(
                            random_state=42, **tree_config
                        )
                    else:
                        new_estimator = estimator
                else:
                    new_estimator = estimator  # Keep non-tree estimators unchanged

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
                    "depth": (
                        str(config["max_depth"])
                        if config["max_depth"] is not None
                        else "None"
                    ),
                    "features": str(config["max_features"])[:6],
                    "split": config["min_samples_split"],
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    return opts, max_acc, best_f1


def _optimize_voting(X_train, y_train, X_test, y_true, cycles=2):
    """
    Optimizes the hyperparameters for a VotingClassifier.
    :param X_train: Training data
    :param y_train: Training labels
    :param X_test: Test data
    :param y_true: True labels for the test data
    :param cycles: Number of optimization cycles
    """

    # Initialize default parameters
    opts = {
        "voting": {
            "voting": "soft",
            "n_jobs": -1,
            "flatten_transform": True,
            "verbose": 0,
        },
        "estimators": [],  # Will be set during optimization
    }

    # Track results
    ma_vec = []
    f1_vec = []

    # Main optimization loop with overall progress bar
    with tqdm(
        range(cycles), desc="Voting Classifier Optimization Cycles", position=0
    ) as cycle_pbar:
        for c in cycle_pbar:
            cycle_pbar.set_description(f"Voting Cycle {c + 1}/{cycles}")

            # Core optimizations
            opts, _, _ = _optimize_estimator_combinations(
                X_train, y_train, X_test, y_true, opts
            )
            opts, _, _ = _optimize_voting_config(X_train, y_train, X_test, y_true, opts)
            opts, _, _ = _optimize_ensemble_sizes(
                X_train, y_train, X_test, y_true, opts
            )
            opts, ma, f1 = _optimize_tree_parameters(
                X_train, y_train, X_test, y_true, opts
            )

            ma_vec.append(ma)
            f1_vec.append(f1)

            # Display progress
            estimator_names = [name for name, _ in opts["estimators"]]
            cycle_pbar.set_postfix(
                {
                    "accuracy": f"{ma:.4f}",
                    "f1": f"{f1:.4f}",
                    "best_overall": f"{max(ma_vec):.4f}",
                    "estimators": "+".join(estimator_names)[:15],
                    "voting": opts["voting"]["voting"],
                    "n_estimators": len(opts["estimators"]),
                    "flatten": "Y" if opts["voting"]["flatten_transform"] else "N",
                }
            )

    return opts, ma, f1, ma_vec, f1_vec


def _analyze_voting_performance(X_train, y_train, X_test, y_true, best_opts):
    """Analyze the performance of voting vs individual estimators"""

    print("\n" + "=" * 70)
    print("VOTING CLASSIFIER PERFORMANCE ANALYSIS")
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
    print(f"Individual Estimator Performance:")
    individual_performances = []

    for name, estimator in best_opts["estimators"]:
        estimator.fit(X_train, y_train)
        y_pred_individual = estimator.predict(X_test)
        acc_individual = accuracy_score(y_true, y_pred_individual)
        f1_individual = f1_score(y_true, y_pred_individual, average="weighted")
        individual_performances.append((name, acc_individual, f1_individual))
        print(f"  {name:15s}: Accuracy={acc_individual:.4f}, F1={f1_individual:.4f}")

    # Voting performance
    print(f"\nVoting Classifier:")
    print(f"  Voting Method: {best_opts['voting']['voting']}")
    print(f"  Flatten Transform: {best_opts['voting']['flatten_transform']}")
    print(f"  Accuracy: {acc_voting:.4f}")
    print(f"  F1 Score: {f1_voting:.4f}")

    # Best individual estimator
    best_individual = max(individual_performances, key=lambda x: x[1])
    best_name, best_acc, best_f1 = best_individual

    print(f"\nBest Individual Estimator: {best_name}")
    print(f"  Accuracy: {best_acc:.4f}")
    print(f"  F1 Score: {best_f1:.4f}")

    # Improvement analysis
    acc_improvement = acc_voting - best_acc
    f1_improvement = f1_voting - best_f1

    print(f"\nVoting Improvement over Best Individual:")
    print(
        f"  Accuracy: {acc_improvement:+.4f} ({acc_improvement / best_acc * 100:+.1f}%)"
    )
    print(f"  F1 Score: {f1_improvement:+.4f} ({f1_improvement / best_f1 * 100:+.1f}%)")

    # Average performance
    avg_acc = np.mean([perf[1] for perf in individual_performances])
    avg_f1 = np.mean([perf[2] for perf in individual_performances])

    print(f"\nVoting vs Average Individual Performance:")
    print(
        f"  Accuracy: {acc_voting:.4f} vs {avg_acc:.4f} (avg) = {acc_voting - avg_acc:+.4f}"
    )
    print(
        f"  F1 Score: {f1_voting:.4f} vs {avg_f1:.4f} (avg) = {f1_voting - avg_f1:+.4f}"
    )

    # Ensemble diversity analysis
    print(f"\nEnsemble Composition:")
    print(f"  Number of Estimators: {len(best_opts['estimators'])}")

    # Count estimator types
    estimator_types = {}
    for name, estimator in best_opts["estimators"]:
        est_type = type(estimator).__name__
        estimator_types[est_type] = estimator_types.get(est_type, 0) + 1

    for est_type, count in estimator_types.items():
        print(f"  {est_type}: {count}")

    # Prediction agreement analysis
    if best_opts["voting"]["voting"] == "soft" and len(best_opts["estimators"]) > 1:
        print(f"\nPrediction Analysis:")
        agreements = 0
        total_predictions = len(y_true)

        # Get individual predictions
        individual_preds = []
        for name, estimator in best_opts["estimators"]:
            individual_preds.append(estimator.predict(X_test))

        # Calculate agreement
        for i in range(total_predictions):
            pred_set = set(pred[i] for pred in individual_preds)
            if len(pred_set) == 1:  # All estimators agree
                agreements += 1

        agreement_rate = agreements / total_predictions
        print(f"  Estimator Agreement Rate: {agreement_rate:.1%}")
        print(f"  Disagreement Rate: {1 - agreement_rate:.1%}")

    return {
        "voting_accuracy": acc_voting,
        "voting_f1": f1_voting,
        "best_individual_accuracy": best_acc,
        "best_individual_f1": best_f1,
        "accuracy_improvement": acc_improvement,
        "f1_improvement": f1_improvement,
        "individual_performances": individual_performances,
        "average_accuracy": avg_acc,
        "average_f1": avg_f1,
    }


# Example usage function
def example_usage():
    """Example of how to use the optimized Voting Classifier function"""
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
    print("Starting Voting Classifier optimization...")

    # Run optimization
    best_opts, best_acc, best_f1, acc_history, f1_history = _optimize_voting_classifier(
        X_train_scaled, y_train, X_test_scaled, y_test, cycles=2
    )

    print(f"\nOptimization completed!")
    print(f"Best accuracy: {best_acc:.4f}")
    print(f"Best F1 score: {best_f1:.4f}")

    # Analyze performance
    analysis = _analyze_voting_performance(
        X_train_scaled, y_train, X_test_scaled, y_test, best_opts
    )

    return best_opts, best_acc, best_f1, acc_history, f1_history


if __name__ == "__main__":
    # Run example
    example_usage()
