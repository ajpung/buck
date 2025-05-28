from typing import Any
import warnings
import numpy as np
from tqdm.auto import tqdm
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.exceptions import ConvergenceWarning

# Suppress convergence warnings for cleaner progress bars
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def _safe_evaluate_model(X_train, y_train, X_test, y_true, **kwargs):
    """Safely evaluate a model configuration"""
    try:
        classifier = SelfTrainingClassifier(**kwargs)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        return accuracy, f1, True
    except Exception:
        return 0.0, 0.0, False


def _optimize_base_estimator(X_train, y_train, X_test, y_true, opts):
    """Optimize the base estimator (most important parameter)"""
    max_acc = -np.inf
    best_f1 = 0.0

    # Different base estimators for self-training
    estimators = [
        ("LogisticRegression", LogisticRegression(random_state=42, max_iter=1000)),
        ("RandomForest", RandomForestClassifier(n_estimators=50, random_state=42)),
        (
            "SVM_RBF",
            SVC(probability=True, random_state=42),
        ),  # probability=True needed for self-training
        ("SVM_Linear", SVC(kernel="linear", probability=True, random_state=42)),
        ("GaussianNB", GaussianNB()),
        ("DecisionTree", DecisionTreeClassifier(random_state=42)),
    ]

    best_estimator = estimators[0][1]
    best_name = estimators[0][0]

    with tqdm(estimators, desc="Optimizing Base Estimator", leave=False) as pbar:
        for name, estimator in pbar:
            test_opts = opts.copy()
            test_opts["estimator"] = estimator

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, **test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_estimator = estimator
                best_name = name

            pbar.set_postfix(
                {
                    "estimator": name[:10],
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                    "best_est": best_name[:8],
                }
            )

    opts["estimator"] = best_estimator
    opts["estimator_name"] = best_name  # Store for display purposes
    return opts, max_acc, best_f1


def _optimize_criterion_config(X_train, y_train, X_test, y_true, opts):
    """Optimize criterion configuration (criterion + threshold/k_best together)"""
    max_acc = -np.inf
    best_f1 = 0.0

    # Combined criterion configurations for efficiency
    configs = [
        {"criterion": "threshold", "threshold": 0.6, "k_best": 10},
        {"criterion": "threshold", "threshold": 0.7, "k_best": 10},
        {"criterion": "threshold", "threshold": 0.75, "k_best": 10},
        {"criterion": "threshold", "threshold": 0.8, "k_best": 10},
        {"criterion": "threshold", "threshold": 0.9, "k_best": 10},
        {"criterion": "k_best", "threshold": 0.75, "k_best": 5},
        {"criterion": "k_best", "threshold": 0.75, "k_best": 10},
        {"criterion": "k_best", "threshold": 0.75, "k_best": 20},
        {"criterion": "k_best", "threshold": 0.75, "k_best": 50},
    ]
    best_config = configs[0]

    with tqdm(configs, desc="Optimizing Criterion Config", leave=False) as pbar:
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
                    "criterion": config["criterion"][:8],
                    "threshold": f"{config['threshold']:.2f}",
                    "k_best": config["k_best"],
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts.update(best_config)
    return opts, max_acc, best_f1


def _optimize_max_iter(X_train, y_train, X_test, y_true, opts):
    """Optimize maximum iterations"""
    max_acc = -np.inf
    best_f1 = 0.0
    # Strategic iteration values instead of linear range
    variable_array = [5, 10, 20, 50, 100, 200]
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


def _create_semi_supervised_data(X, y, labeled_ratio=0.3):
    """Create semi-supervised dataset by masking some labels"""
    n_samples = len(y)
    n_labeled = int(n_samples * labeled_ratio)

    # Randomly select labeled samples
    labeled_indices = np.random.choice(n_samples, n_labeled, replace=False)

    # Create semi-supervised labels (-1 for unlabeled)
    y_semi = np.full_like(y, -1)
    y_semi[labeled_indices] = y[labeled_indices]

    return X, y_semi, labeled_indices


def _optimize_selftrain(X_train, y_train, X_test, y_true, cycles=2, labeled_ratio=0.3):
    """
    Optimizes the hyperparameters for SelfTrainingClassifier.
    :param X_train: Training data
    :param y_train: Training labels (will be partially masked for semi-supervised learning)
    :param X_test: Test data
    :param y_true: True labels for the test data
    :param cycles: Number of optimization cycles
    :param labeled_ratio: Fraction of training data to keep labeled
    """

    # Create semi-supervised training data
    X_train_semi, y_train_semi, labeled_indices = _create_semi_supervised_data(
        X_train, y_train, labeled_ratio
    )

    print(
        f"Semi-supervised setup: {len(labeled_indices)} labeled samples out of {len(y_train)} total"
    )
    print(f"Labeled ratio: {labeled_ratio:.1%}")

    # Define initial parameters
    opts = {
        "estimator": LogisticRegression(random_state=42, max_iter=1000),
        "threshold": 0.75,
        "criterion": "threshold",
        "k_best": 10,
        "max_iter": 10,
        "verbose": False,
        "estimator_name": "LogisticRegression",  # For display
    }

    # Track results
    ma_vec = []
    f1_vec = []

    # Main optimization loop with overall progress bar
    with tqdm(
        range(cycles), desc="Self-Training Optimization Cycles", position=0
    ) as cycle_pbar:
        for c in cycle_pbar:
            cycle_pbar.set_description(f"Self-Training Cycle {c + 1}/{cycles}")

            # Core hyperparameters
            opts, _, _ = _optimize_base_estimator(
                X_train_semi, y_train_semi, X_test, y_true, opts
            )
            opts, _, _ = _optimize_criterion_config(
                X_train_semi, y_train_semi, X_test, y_true, opts
            )
            opts, ma, f1 = _optimize_max_iter(
                X_train_semi, y_train_semi, X_test, y_true, opts
            )

            ma_vec.append(ma)
            f1_vec.append(f1)

            cycle_pbar.set_postfix(
                {
                    "accuracy": f"{ma:.4f}",
                    "f1": f"{f1:.4f}",
                    "best_overall": f"{max(ma_vec):.4f}",
                    "estimator": opts.get("estimator_name", "Unknown")[:8],
                    "criterion": opts["criterion"][:8],
                    "threshold": f"{opts['threshold']:.2f}",
                    "k_best": opts["k_best"],
                    "max_iter": opts["max_iter"],
                }
            )

    return opts, ma, f1, ma_vec, f1_vec, X_train_semi, y_train_semi, labeled_indices


def _analyze_self_training_performance(
    X_train_semi, y_train_semi, X_test, y_true, best_opts, labeled_indices
):
    """Analyze the performance of self-training vs base estimator"""

    print("\n" + "=" * 60)
    print("SELF-TRAINING PERFORMANCE ANALYSIS")
    print("=" * 60)

    # Train self-training classifier
    self_training_clf = SelfTrainingClassifier(
        **{k: v for k, v in best_opts.items() if k != "estimator_name"}
    )
    self_training_clf.fit(X_train_semi, y_train_semi)

    # Train base estimator on labeled data only
    base_estimator = best_opts["estimator"]
    X_labeled = X_train_semi[labeled_indices]
    y_labeled = y_train_semi[labeled_indices]
    base_estimator.fit(X_labeled, y_labeled)

    # Make predictions
    y_pred_self_training = self_training_clf.predict(X_test)
    y_pred_base = base_estimator.predict(X_test)

    # Calculate metrics
    acc_self_training = accuracy_score(y_true, y_pred_self_training)
    f1_self_training = f1_score(y_true, y_pred_self_training, average="weighted")

    acc_base = accuracy_score(y_true, y_pred_base)
    f1_base = f1_score(y_true, y_pred_base, average="weighted")

    # Display results
    print(
        f"Base Estimator ({best_opts.get('estimator_name', 'Unknown')}) - Labeled Data Only:"
    )
    print(f"  Accuracy: {acc_base:.4f}")
    print(f"  F1 Score: {f1_base:.4f}")
    print(f"  Training Samples: {len(labeled_indices)}")

    print(f"\nSelf-Training Classifier:")
    print(f"  Accuracy: {acc_self_training:.4f}")
    print(f"  F1 Score: {f1_self_training:.4f}")
    print(f"  Initial Labeled Samples: {len(labeled_indices)}")
    print(f"  Total Training Samples: {len(y_train_semi)}")

    # Calculate improvement
    acc_improvement = acc_self_training - acc_base
    f1_improvement = f1_self_training - f1_base

    print(f"\nImprovement from Self-Training:")
    print(
        f"  Accuracy: {acc_improvement:+.4f} ({acc_improvement / acc_base * 100:+.1f}%)"
    )
    print(f"  F1 Score: {f1_improvement:+.4f} ({f1_improvement / f1_base * 100:+.1f}%)")

    # Analyze self-training iterations
    if hasattr(self_training_clf, "n_iter_"):
        print(f"\nSelf-Training Details:")
        print(f"  Iterations completed: {self_training_clf.n_iter_}")
        print(f"  Termination criterion: {best_opts['criterion']}")
        if best_opts["criterion"] == "threshold":
            print(f"  Confidence threshold: {best_opts['threshold']}")
        else:
            print(f"  K-best per iteration: {best_opts['k_best']}")

    return {
        "self_training_accuracy": acc_self_training,
        "self_training_f1": f1_self_training,
        "base_accuracy": acc_base,
        "base_f1": f1_base,
        "accuracy_improvement": acc_improvement,
        "f1_improvement": f1_improvement,
    }


# Example usage function
def example_usage():
    """Example of how to use the optimized Self-Training function"""
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
    print("Starting Self-Training optimization...")

    # Run optimization with different labeled ratios
    labeled_ratios = [0.2, 0.3, 0.5]

    for labeled_ratio in labeled_ratios:
        print(f"\n{'=' * 60}")
        print(f"TESTING WITH {labeled_ratio:.0%} LABELED DATA")
        print(f"{'=' * 60}")

        # Run optimization
        results = _optimize_self_training(
            X_train_scaled,
            y_train,
            X_test_scaled,
            y_test,
            cycles=2,
            labeled_ratio=labeled_ratio,
        )
        (
            best_opts,
            best_acc,
            best_f1,
            acc_history,
            f1_history,
            X_train_semi,
            y_train_semi,
            labeled_indices,
        ) = results

        print(f"\nOptimization completed for {labeled_ratio:.0%} labeled data!")
        print(f"Best accuracy: {best_acc:.4f}")
        print(f"Best F1 score: {best_f1:.4f}")
        print(f"Best parameters:")
        for param, value in best_opts.items():
            if param != "estimator":  # Don't print the full estimator object
                print(f"  {param}: {value}")

        # Analyze performance
        analysis = _analyze_self_training_performance(
            X_train_semi,
            y_train_semi,
            X_test_scaled,
            y_test,
            best_opts,
            labeled_indices,
        )

    return best_opts, best_acc, best_f1, acc_history, f1_history


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    # Run example
    example_usage()
