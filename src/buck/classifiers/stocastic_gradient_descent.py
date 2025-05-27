from typing import Any
import warnings
import numpy as np
from tqdm.auto import tqdm
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.exceptions import ConvergenceWarning

# Suppress convergence warnings for cleaner progress bars
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def _safe_evaluate_model(X_train, y_train, X_test, y_true, **kwargs):
    """Safely evaluate a model configuration"""
    try:
        classifier = SGDClassifier(**kwargs)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        return accuracy, f1, True
    except Exception:
        return 0.0, 0.0, False


def _optimize_loss_penalty_combo(X_train, y_train, X_test, y_true, opts):
    """Optimize loss function and penalty together for efficiency"""
    max_acc = -np.inf
    best_f1 = 0.0

    # Strategic loss-penalty combinations
    combinations = [
        # SVM-like approaches
        {"loss": "hinge", "penalty": "l2"},
        {"loss": "hinge", "penalty": "l1"},
        {"loss": "squared_hinge", "penalty": "l2"},
        # Logistic regression-like approaches
        {"loss": "log_loss", "penalty": "l2"},
        {"loss": "log_loss", "penalty": "l1"},
        {"loss": "log_loss", "penalty": "elasticnet"},
        # Robust approaches
        {"loss": "modified_huber", "penalty": "l2"},
        {"loss": "modified_huber", "penalty": "l1"},
        # Perceptron
        {"loss": "perceptron", "penalty": "l2"},
    ]

    best_combo = combinations[0]

    with tqdm(combinations, desc="Optimizing Loss-Penalty Combo", leave=False) as pbar:
        for combo in pbar:
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
    """Optimize regularization strength (alpha)"""
    max_acc = -np.inf
    best_f1 = 0.0
    # Strategic alpha values - reduced from 15 to 8 for efficiency
    variable_array = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
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
                    "alpha": f"{v:.0e}",
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts["alpha"] = best_val
    return opts, max_acc, best_f1


def _optimize_learning_rate_config(X_train, y_train, X_test, y_true, opts):
    """Optimize learning rate configuration (schedule + eta0 + power_t together)"""
    max_acc = -np.inf
    best_f1 = 0.0

    # Combined learning rate configurations
    configs = [
        {
            "learning_rate": "optimal",
            "eta0": 0.0,
            "power_t": 0.5,
        },  # eta0 ignored for optimal
        {"learning_rate": "constant", "eta0": 0.01, "power_t": 0.5},
        {"learning_rate": "constant", "eta0": 0.1, "power_t": 0.5},
        {"learning_rate": "constant", "eta0": 1.0, "power_t": 0.5},
        {"learning_rate": "invscaling", "eta0": 0.01, "power_t": 0.25},
        {"learning_rate": "invscaling", "eta0": 0.01, "power_t": 0.5},
        {"learning_rate": "invscaling", "eta0": 0.1, "power_t": 0.5},
        {"learning_rate": "adaptive", "eta0": 0.01, "power_t": 0.5},
        {"learning_rate": "adaptive", "eta0": 0.1, "power_t": 0.5},
    ]
    best_config = configs[0]

    with tqdm(configs, desc="Optimizing Learning Rate Config", leave=False) as pbar:
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
                    "lr_sched": config["learning_rate"][:6],
                    "eta0": f"{config['eta0']:.3f}",
                    "power_t": f"{config['power_t']:.2f}",
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts.update(best_config)
    return opts, max_acc, best_f1


def _optimize_l1_ratio(X_train, y_train, X_test, y_true, opts):
    """Optimize L1 ratio for ElasticNet penalty"""
    max_acc = -np.inf
    best_f1 = 0.0

    # Only optimize if penalty is elasticnet
    if opts["penalty"] != "elasticnet":
        return opts, max_acc, best_f1

    # Strategic L1 ratios
    variable_array = [0.0, 0.15, 0.3, 0.5, 0.7, 0.85, 1.0]
    best_val = variable_array[0]

    with tqdm(variable_array, desc="Optimizing L1 Ratio", leave=False) as pbar:
        for v in pbar:
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
    """Optimize convergence configuration (max_iter + tol + early_stopping together)"""
    max_acc = -np.inf
    best_f1 = 0.0

    # Combined convergence configurations
    configs = [
        {"max_iter": 1000, "tol": 1e-3, "early_stopping": False},
        {"max_iter": 2000, "tol": 1e-3, "early_stopping": False},
        {"max_iter": 5000, "tol": 1e-3, "early_stopping": False},
        {"max_iter": 2000, "tol": 1e-4, "early_stopping": False},
        {"max_iter": 5000, "tol": 1e-4, "early_stopping": False},
        {
            "max_iter": 10000,
            "tol": 1e-3,
            "early_stopping": True,
            "validation_fraction": 0.1,
            "n_iter_no_change": 5,
        },
        {
            "max_iter": 10000,
            "tol": 1e-4,
            "early_stopping": True,
            "validation_fraction": 0.1,
            "n_iter_no_change": 10,
        },
        {
            "max_iter": 20000,
            "tol": 1e-4,
            "early_stopping": True,
            "validation_fraction": 0.2,
            "n_iter_no_change": 10,
        },
    ]
    best_config = configs[0]

    with tqdm(configs, desc="Optimizing Convergence Config", leave=False) as pbar:
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
                    "max_iter": config["max_iter"],
                    "tol": f"{config['tol']:.0e}",
                    "early_stop": "Y" if config["early_stopping"] else "N",
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
        {"fit_intercept": True, "shuffle": True, "average": False, "warm_start": False},
        {"fit_intercept": True, "shuffle": True, "average": True, "warm_start": False},
        {
            "fit_intercept": True,
            "shuffle": False,
            "average": False,
            "warm_start": False,
        },
        {
            "fit_intercept": False,
            "shuffle": True,
            "average": False,
            "warm_start": False,
        },
        {"fit_intercept": True, "shuffle": True, "average": False, "warm_start": True},
        {"fit_intercept": False, "shuffle": True, "average": True, "warm_start": False},
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
                    "warm_start": "Y" if config["warm_start"] else "N",
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts.update(best_config)
    return opts, max_acc, best_f1


def _optimize_class_weight(X_train, y_train, X_test, y_true, opts):
    """Optimize class weights"""
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


def _optimize_sgd_classifier(X_train, y_train, X_test, y_true, cycles=2):
    """
    Optimizes the hyperparameters for SGDClassifier.
    :param X_train: Training data
    :param y_train: Training labels
    :param X_test: Test data
    :param y_true: True labels for the test data
    :param cycles: Number of optimization cycles
    """

    # Define initial parameters with better defaults
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

    # Main optimization loop with overall progress bar
    with tqdm(
        range(cycles), desc="SGD Classifier Optimization Cycles", position=0
    ) as cycle_pbar:
        for c in cycle_pbar:
            cycle_pbar.set_description(f"SGD Cycle {c + 1}/{cycles}")

            # Core hyperparameters (most impactful for SGD)
            opts, _, _ = _optimize_loss_penalty_combo(
                X_train, y_train, X_test, y_true, opts
            )
            opts, _, _ = _optimize_alpha(X_train, y_train, X_test, y_true, opts)
            opts, _, _ = _optimize_learning_rate_config(
                X_train, y_train, X_test, y_true, opts
            )

            # Conditional optimizations
            opts, _, _ = _optimize_l1_ratio(X_train, y_train, X_test, y_true, opts)

            # Convergence and other parameters
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
                    "loss": opts["loss"][:6],
                    "penalty": opts["penalty"][:6] if opts["penalty"] else "None",
                    "alpha": f"{opts['alpha']:.0e}",
                    "lr_sched": opts["learning_rate"][:6],
                    "max_iter": opts["max_iter"],
                }
            )

    return opts, ma, f1, ma_vec, f1_vec


def _analyze_sgd_performance(X_train, y_train, X_test, y_true, best_opts):
    """Analyze SGD performance and convergence characteristics"""

    print("\n" + "=" * 60)
    print("SGD CLASSIFIER PERFORMANCE ANALYSIS")
    print("=" * 60)

    # Train final model with verbose output
    analysis_opts = best_opts.copy()
    analysis_opts["verbose"] = 1

    print("Training final SGD model with best parameters...")
    sgd_clf = SGDClassifier(**analysis_opts)
    sgd_clf.fit(X_train, y_train)

    # Make predictions
    y_pred = sgd_clf.predict(X_test)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")

    print(f"\nFinal Model Performance:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 Score: {f1:.4f}")

    # Analyze configuration
    print(f"\nOptimal Configuration:")
    print(f"  Loss Function: {best_opts['loss']}")
    print(f"  Penalty: {best_opts['penalty']}")
    print(f"  Alpha (regularization): {best_opts['alpha']:.0e}")

    if best_opts["penalty"] == "elasticnet":
        print(f"  L1 Ratio: {best_opts['l1_ratio']:.2f}")

    print(f"  Learning Rate Schedule: {best_opts['learning_rate']}")
    if best_opts["learning_rate"] != "optimal":
        print(f"  Initial Learning Rate (eta0): {best_opts['eta0']:.4f}")
    if best_opts["learning_rate"] == "invscaling":
        print(f"  Power T: {best_opts['power_t']:.2f}")

    print(f"  Max Iterations: {best_opts['max_iter']}")
    print(f"  Tolerance: {best_opts['tol']:.0e}")
    print(f"  Early Stopping: {best_opts['early_stopping']}")

    if best_opts["early_stopping"]:
        print(f"  Validation Fraction: {best_opts['validation_fraction']:.2f}")
        print(f"  N Iter No Change: {best_opts['n_iter_no_change']}")

    print(f"  Fit Intercept: {best_opts['fit_intercept']}")
    print(f"  Shuffle: {best_opts['shuffle']}")
    print(f"  Average: {best_opts['average']}")
    print(f"  Class Weight: {best_opts['class_weight']}")

    # Convergence analysis
    if hasattr(sgd_clf, "n_iter_"):
        print(f"\nConvergence Analysis:")
        print(f"  Actual iterations: {sgd_clf.n_iter_}")
        print(
            f"  Converged: {'Yes' if sgd_clf.n_iter_ < best_opts['max_iter'] else 'No'}"
        )

    # Loss function interpretation
    print(f"\nLoss Function Interpretation:")
    loss_descriptions = {
        "hinge": "SVM-like, good for linearly separable data",
        "log_loss": "Logistic regression, provides probabilities",
        "modified_huber": "Robust to outliers, provides probabilities",
        "squared_hinge": "Differentiable version of hinge",
        "perceptron": "Simple linear classifier",
    }
    print(
        f"  {best_opts['loss']}: {loss_descriptions.get(best_opts['loss'], 'Unknown')}"
    )

    # Penalty interpretation
    penalty_descriptions = {
        "l2": "Ridge regularization, keeps all features",
        "l1": "Lasso regularization, promotes sparsity",
        "elasticnet": "Combines L1 and L2 regularization",
        None: "No regularization",
    }
    print(
        f"  {best_opts['penalty']}: {penalty_descriptions.get(best_opts['penalty'], 'Unknown')}"
    )

    return {
        "accuracy": accuracy,
        "f1": f1,
        "n_iter": getattr(sgd_clf, "n_iter_", None),
        "converged": (
            sgd_clf.n_iter_ < best_opts["max_iter"]
            if hasattr(sgd_clf, "n_iter_")
            else None
        ),
    }


# Example usage function
def example_usage():
    """Example of how to use the optimized SGD Classifier function"""
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # Generate sample data
    print("Generating sample classification data...")
    X, y = make_classification(
        n_samples=5000,  # Larger dataset for SGD
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

    # Scale the data (very important for SGD)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Training set: {X_train_scaled.shape}")
    print(f"Test set: {X_test_scaled.shape}")
    print("Starting SGD Classifier optimization...")

    # Run optimization
    best_opts, best_acc, best_f1, acc_history, f1_history = _optimize_sgd_classifier(
        X_train_scaled, y_train, X_test_scaled, y_test, cycles=2
    )

    print(f"\nOptimization completed!")
    print(f"Best accuracy: {best_acc:.4f}")
    print(f"Best F1 score: {best_f1:.4f}")

    # Analyze performance
    analysis = _analyze_sgd_performance(
        X_train_scaled, y_train, X_test_scaled, y_test, best_opts
    )

    return best_opts, best_acc, best_f1, acc_history, f1_history


if __name__ == "__main__":
    # Run example
    example_usage()
