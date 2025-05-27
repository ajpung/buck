from typing import Any
import warnings
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score

# Handle CatBoost import with fallback
try:
    from catboost import CatBoostClassifier

    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not available. Install with: pip install catboost")

# Suppress warnings for cleaner progress bars
warnings.filterwarnings("ignore")


def _safe_evaluate_model(X_train, y_train, X_test, y_true, **kwargs):
    """Safely evaluate a CatBoost model configuration"""
    if not CATBOOST_AVAILABLE:
        return 0.0, 0.0, False

    try:
        # CatBoost specific parameters
        catboost_params = kwargs.copy()
        catboost_params["verbose"] = False  # Always suppress training output
        catboost_params["allow_writing_files"] = False  # Don't write temp files

        classifier = CatBoostClassifier(**catboost_params)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        return accuracy, f1, True
    except Exception:
        return 0.0, 0.0, False


def _optimize_boosting_config(X_train, y_train, X_test, y_true, opts):
    """Optimize boosting configuration (iterations + learning_rate together)"""
    max_acc = -np.inf
    best_f1 = 0.0

    # Strategic boosting configurations for CatBoost
    configs = [
        {"iterations": 500, "learning_rate": 0.1},  # Default balanced
        {"iterations": 1000, "learning_rate": 0.05},  # Many trees, slower learning
        {"iterations": 1500, "learning_rate": 0.03},  # Many trees, very slow learning
        {"iterations": 300, "learning_rate": 0.15},  # Fewer trees, faster learning
        {"iterations": 200, "learning_rate": 0.2},  # Few trees, aggressive learning
        {"iterations": 750, "learning_rate": 0.08},  # Moderate compromise
        {"iterations": 100, "learning_rate": 0.3},  # Very few trees, very aggressive
        {"iterations": 2000, "learning_rate": 0.01},  # Many trees, very conservative
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
                    "iter": config["iterations"],
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

    # Strategic tree structure configurations for CatBoost
    configs = [
        {"depth": 6, "min_data_in_leaf": 1, "grow_policy": "SymmetricTree"},  # Default
        {
            "depth": 4,
            "min_data_in_leaf": 1,
            "grow_policy": "SymmetricTree",
        },  # Shallow trees
        {
            "depth": 8,
            "min_data_in_leaf": 1,
            "grow_policy": "SymmetricTree",
        },  # Deeper trees
        {
            "depth": 10,
            "min_data_in_leaf": 1,
            "grow_policy": "SymmetricTree",
        },  # Very deep
        {
            "depth": 6,
            "min_data_in_leaf": 5,
            "grow_policy": "SymmetricTree",
        },  # Conservative splits
        {
            "depth": 6,
            "min_data_in_leaf": 10,
            "grow_policy": "SymmetricTree",
        },  # More conservative
        {
            "depth": 6,
            "min_data_in_leaf": 1,
            "grow_policy": "Lossguide",
        },  # Alternative grow policy
        {
            "depth": 8,
            "min_data_in_leaf": 3,
            "grow_policy": "Depthwise",
        },  # Depthwise growth
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

            pbar.set_postfix(
                {
                    "depth": config["depth"],
                    "min_leaf": config["min_data_in_leaf"],
                    "policy": config["grow_policy"][:6],
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

    # Strategic regularization configurations for CatBoost
    configs = [
        {"l2_leaf_reg": 3, "reg_lambda": None},  # Default L2
        {"l2_leaf_reg": 1, "reg_lambda": None},  # Light L2
        {"l2_leaf_reg": 10, "reg_lambda": None},  # Strong L2
        {"l2_leaf_reg": 3, "reg_lambda": 1},  # L2 + lambda
        {"l2_leaf_reg": 5, "reg_lambda": 0.5},  # Balanced regularization
        {"l2_leaf_reg": 0.1, "reg_lambda": None},  # Very light L2
        {"l2_leaf_reg": 20, "reg_lambda": None},  # Very strong L2
        {"l2_leaf_reg": 3, "reg_lambda": 5},  # Strong lambda
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

            lambda_str = (
                f"{config['reg_lambda']:.1f}"
                if config["reg_lambda"] is not None
                else "None"
            )
            pbar.set_postfix(
                {
                    "l2_leaf": f"{config['l2_leaf_reg']:.1f}",
                    "lambda": lambda_str,
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

    # Strategic sampling configurations for CatBoost
    configs = [
        {"subsample": 1.0, "rsm": 1.0},  # No sampling
        {"subsample": 0.8, "rsm": 0.8},  # Moderate sampling
        {"subsample": 0.9, "rsm": 0.9},  # Light sampling
        {"subsample": 0.7, "rsm": 0.7},  # Aggressive sampling
        {"subsample": 0.8, "rsm": 1.0},  # Row sampling only
        {"subsample": 1.0, "rsm": 0.8},  # Feature sampling only
        {"subsample": 0.6, "rsm": 0.8},  # Heavy row sampling
        {"subsample": 0.8, "rsm": 0.6},  # Heavy feature sampling
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
                    "rsm": f"{config['rsm']:.1f}",
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts.update(best_config)
    return opts, max_acc, best_f1


def _optimize_loss_function(X_train, y_train, X_test, y_true, opts):
    """Optimize loss function and related parameters"""
    max_acc = -np.inf
    best_f1 = 0.0

    # Determine if binary or multi-class
    n_classes = len(np.unique(y_train))

    if n_classes == 2:  # Binary classification
        configs = [
            {"loss_function": "Logloss"},
            {"loss_function": "CrossEntropy"},
        ]
    else:  # Multi-class classification
        configs = [
            {"loss_function": "MultiClass"},
            {"loss_function": "MultiClassOneVsAll"},
        ]

    best_config = configs[0]

    with tqdm(configs, desc="Optimizing Loss Function", leave=False) as pbar:
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
                    "loss": config["loss_function"][:8],
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts.update(best_config)
    return opts, max_acc, best_f1


def _optimize_bootstrap_config(X_train, y_train, X_test, y_true, opts):
    """Optimize bootstrap configuration"""
    max_acc = -np.inf
    best_f1 = 0.0

    # Bootstrap configurations for CatBoost
    configs = [
        {"bootstrap_type": "Bayesian"},  # Default for small datasets
        {"bootstrap_type": "Bernoulli", "subsample": 0.66},  # Bernoulli bootstrap
        {"bootstrap_type": "MVS"},  # Minimum variance sampling
        {"bootstrap_type": "Poisson"},  # Poisson bootstrap
        {"bootstrap_type": "No"},  # No bootstrap
    ]

    # For large datasets, prefer faster options
    n_samples = X_train.shape[0]
    if n_samples > 10000:
        configs = [
            {"bootstrap_type": "MVS"},
            {"bootstrap_type": "Bernoulli", "subsample": 0.8},
            {"bootstrap_type": "No"},
        ]

    best_config = configs[0]

    with tqdm(configs, desc="Optimizing Bootstrap Config", leave=False) as pbar:
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
                    "bootstrap": config["bootstrap_type"][:6],
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts.update(best_config)
    return opts, max_acc, best_f1


def _optimize_catboost_classifier(X_train, y_train, X_test, y_true, cycles=2):
    """
    Optimizes the hyperparameters for CatBoost Classifier.
    :param X_train: Training data
    :param y_train: Training labels
    :param X_test: Test data
    :param y_true: True labels for the test data
    :param cycles: Number of optimization cycles
    """

    if not CATBOOST_AVAILABLE:
        print("CatBoost is not available. Please install it with: pip install catboost")
        return {}, 0.0, 0.0, [], []

    # Determine problem type
    n_classes = len(np.unique(y_train))
    n_samples, n_features = X_train.shape

    # Define initial parameters with good defaults
    opts = {
        "iterations": 500,
        "learning_rate": 0.1,
        "depth": 6,
        "l2_leaf_reg": 3,
        "reg_lambda": None,
        "subsample": 1.0,
        "rsm": 1.0,  # Random subspace method (feature sampling)
        "min_data_in_leaf": 1,
        "grow_policy": "SymmetricTree",
        "bootstrap_type": "Bayesian" if n_samples < 10000 else "MVS",
        "loss_function": "MultiClass" if n_classes > 2 else "Logloss",
        "eval_metric": "MultiClass" if n_classes > 2 else "Logloss",
        "random_seed": 42,
        "thread_count": -1,
        "verbose": False,
        "allow_writing_files": False,
        "use_best_model": False,  # Disable early stopping initially
    }

    # Track results
    ma_vec = []
    f1_vec = []

    # Main optimization loop with overall progress bar
    with tqdm(
        range(cycles), desc="CatBoost Optimization Cycles", position=0
    ) as cycle_pbar:
        for c in cycle_pbar:
            cycle_pbar.set_description(f"CatBoost Cycle {c + 1}/{cycles}")

            # Core hyperparameters (most impactful for CatBoost)
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
            opts, _, _ = _optimize_loss_function(X_train, y_train, X_test, y_true, opts)
            opts, ma, f1 = _optimize_bootstrap_config(
                X_train, y_train, X_test, y_true, opts
            )

            ma_vec.append(ma)
            f1_vec.append(f1)

            cycle_pbar.set_postfix(
                {
                    "accuracy": f"{ma:.4f}",
                    "f1": f"{f1:.4f}",
                    "best_overall": f"{max(ma_vec):.4f}",
                    "iter": opts["iterations"],
                    "lr": f"{opts['learning_rate']:.3f}",
                    "depth": opts["depth"],
                    "l2_reg": f"{opts['l2_leaf_reg']:.1f}",
                    "subsample": f"{opts['subsample']:.1f}",
                    "bootstrap": opts["bootstrap_type"][:4],
                }
            )

    return opts, ma, f1, ma_vec, f1_vec


def _analyze_catboost_performance(X_train, y_train, X_test, y_true, best_opts):
    """Analyze CatBoost performance and feature importance"""

    if not CATBOOST_AVAILABLE:
        print("CatBoost is not available for analysis.")
        return {}

    print("\n" + "=" * 60)
    print("CATBOOST CLASSIFIER PERFORMANCE ANALYSIS")
    print("=" * 60)

    # Train final model
    print("Training final CatBoost model with best parameters...")
    final_opts = best_opts.copy()
    final_opts["verbose"] = False
    final_opts["allow_writing_files"] = False

    classifier = CatBoostClassifier(**final_opts)
    classifier.fit(X_train, y_train)

    # Make predictions
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")

    print(f"\nFinal Model Performance:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 Score: {f1:.4f}")

    # Analyze configuration
    print(f"\nOptimal Configuration:")
    print(f"  Iterations: {best_opts['iterations']}")
    print(f"  Learning Rate: {best_opts['learning_rate']:.3f}")
    print(f"  Tree Depth: {best_opts['depth']}")
    print(f"  Min Data in Leaf: {best_opts['min_data_in_leaf']}")
    print(f"  Grow Policy: {best_opts['grow_policy']}")
    print(f"  L2 Leaf Regularization: {best_opts['l2_leaf_reg']:.2f}")

    if best_opts["reg_lambda"] is not None:
        print(f"  Reg Lambda: {best_opts['reg_lambda']:.2f}")

    print(f"  Subsample: {best_opts['subsample']:.2f}")
    print(f"  RSM (Feature Sampling): {best_opts['rsm']:.2f}")
    print(f"  Bootstrap Type: {best_opts['bootstrap_type']}")
    print(f"  Loss Function: {best_opts['loss_function']}")

    # Feature importance analysis
    print(f"\nFeature Importance Analysis:")
    try:
        feature_importance = classifier.get_feature_importance()
        if len(feature_importance) > 0:
            # Get top 5 features
            top_indices = np.argsort(feature_importance)[-5:][::-1]
            print("  Top 5 most important features:")
            for i, idx in enumerate(top_indices):
                print(f"    Feature {idx}: {feature_importance[idx]:.4f}")
        else:
            print("  No feature importance available")
    except:
        print("  Could not compute feature importance")

    # Model characteristics
    print(f"\nModel Characteristics:")
    n_classes = len(np.unique(y_train))
    n_samples, n_features = X_train.shape

    print(
        f"  Problem Type: {'Binary' if n_classes == 2 else 'Multi-class'} Classification"
    )
    print(f"  Number of Classes: {n_classes}")
    print(f"  Training Samples: {n_samples}")
    print(f"  Features: {n_features}")

    # Check for class imbalance
    unique, counts = np.unique(y_train, return_counts=True)
    class_distribution = dict(zip(unique, counts))
    print(f"  Class Distribution: {class_distribution}")

    # Imbalance ratio
    if len(unique) == 2:
        imbalance_ratio = max(counts) / min(counts)
        print(f"  Imbalance Ratio: {imbalance_ratio:.2f}:1")
        if imbalance_ratio > 2:
            print(f"  Dataset is imbalanced - CatBoost handles this well automatically")

    # Bootstrap analysis
    print(f"\nBootstrap Configuration:")
    bootstrap_descriptions = {
        "Bayesian": "Good for small datasets, provides uncertainty estimates",
        "Bernoulli": "Standard bootstrap, good balance of speed and quality",
        "MVS": "Minimum variance sampling, fastest for large datasets",
        "Poisson": "Poisson bootstrap, good for regression-like problems",
        "No": "No bootstrap, fastest but may overfit",
    }
    bootstrap_type = best_opts["bootstrap_type"]
    print(
        f"  {bootstrap_type}: {bootstrap_descriptions.get(bootstrap_type, 'Unknown')}"
    )

    # Regularization analysis
    total_reg = best_opts["l2_leaf_reg"]
    if best_opts["reg_lambda"] is not None:
        total_reg += best_opts["reg_lambda"]

    if total_reg < 1:
        print(f"  Regularization: Light (may be appropriate for small datasets)")
    elif total_reg < 5:
        print(f"  Regularization: Moderate (good balance)")
    else:
        print(f"  Regularization: Strong (good for preventing overfitting)")

    # Sampling analysis
    total_sampling = best_opts["subsample"] * best_opts["rsm"]
    if total_sampling >= 0.9:
        print(f"  Sampling: Minimal (uses most data and features)")
    elif total_sampling >= 0.7:
        print(f"  Sampling: Moderate (good for generalization)")
    else:
        print(f"  Sampling: Aggressive (strong regularization effect)")

    return {
        "accuracy": accuracy,
        "f1": f1,
        "n_classes": n_classes,
        "imbalance_ratio": max(counts) / min(counts) if len(unique) == 2 else 1.0,
        "total_regularization": total_reg,
        "sampling_fraction": total_sampling,
        "bootstrap_type": bootstrap_type,
    }


# Example usage function
def example_usage():
    """Example of how to use the optimized CatBoost Classifier function"""
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    if not CATBOOST_AVAILABLE:
        print("CatBoost is not available. Please install it with: pip install catboost")
        return None, 0.0, 0.0, [], []

    # Generate sample data
    print("Generating sample classification data...")
    X, y = make_classification(
        n_samples=3000,  # Larger dataset for CatBoost
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        weights=[0.5, 0.3, 0.2],  # Imbalanced classes
        random_state=42,
    )

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # CatBoost handles raw features excellently, but we can still scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Training set: {X_train_scaled.shape}")
    print(f"Test set: {X_test_scaled.shape}")
    print("Starting CatBoost Classifier optimization...")

    # Run optimization
    best_opts, best_acc, best_f1, acc_history, f1_history = (
        _optimize_catboost_classifier(
            X_train_scaled, y_train, X_test_scaled, y_test, cycles=2
        )
    )

    print(f"\nOptimization completed!")
    print(f"Best accuracy: {best_acc:.4f}")
    print(f"Best F1 score: {best_f1:.4f}")

    # Analyze performance
    analysis = _analyze_catboost_performance(
        X_train_scaled, y_train, X_test_scaled, y_test, best_opts
    )

    return best_opts, best_acc, best_f1, acc_history, f1_history


if __name__ == "__main__":
    # Run example
    example_usage()
