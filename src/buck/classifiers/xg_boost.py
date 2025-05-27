from typing import Any
import warnings
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score

# Handle XGBoost import with fallback
try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

# Suppress warnings for cleaner progress bars
warnings.filterwarnings("ignore")


def _safe_evaluate_model(X_train, y_train, X_test, y_true, **kwargs):
    """Safely evaluate a model configuration"""
    if not XGBOOST_AVAILABLE:
        return 0.0, 0.0, False

    try:
        # Convert to DMatrix for better performance
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test)

        # Train model
        model = xgb.train(
            kwargs,
            dtrain,
            num_boost_round=kwargs.get("n_estimators", 100),
            verbose_eval=False,
        )

        # Make predictions
        y_pred_proba = model.predict(dtest)

        # Convert probabilities to class predictions
        if len(np.unique(y_train)) == 2:  # Binary classification
            y_pred = (y_pred_proba > 0.5).astype(int)
        else:  # Multi-class classification
            y_pred = np.argmax(
                y_pred_proba.reshape(-1, len(np.unique(y_train))), axis=1
            )

        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        return accuracy, f1, True
    except Exception:
        return 0.0, 0.0, False


def _optimize_boosting_config(X_train, y_train, X_test, y_true, opts):
    """Optimize boosting configuration (n_estimators + learning_rate together)"""
    max_acc = -np.inf
    best_f1 = 0.0

    # Strategic boosting configurations
    configs = [
        {"n_estimators": 100, "learning_rate": 0.1},  # Default balanced
        {"n_estimators": 200, "learning_rate": 0.1},  # More trees, same rate
        {"n_estimators": 300, "learning_rate": 0.05},  # Many trees, slower learning
        {"n_estimators": 500, "learning_rate": 0.03},  # Many trees, very slow learning
        {"n_estimators": 100, "learning_rate": 0.2},  # Fewer trees, faster learning
        {"n_estimators": 150, "learning_rate": 0.15},  # Moderate compromise
        {"n_estimators": 50, "learning_rate": 0.3},  # Few trees, aggressive learning
        {"n_estimators": 1000, "learning_rate": 0.01},  # Many trees, very conservative
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

    # Strategic tree structure configurations
    configs = [
        {"max_depth": 6, "min_child_weight": 1, "gamma": 0},  # Default
        {"max_depth": 4, "min_child_weight": 1, "gamma": 0},  # Shallow trees
        {"max_depth": 8, "min_child_weight": 1, "gamma": 0},  # Deeper trees
        {"max_depth": 6, "min_child_weight": 3, "gamma": 0},  # Conservative splits
        {"max_depth": 6, "min_child_weight": 1, "gamma": 0.1},  # Regularized
        {"max_depth": 4, "min_child_weight": 5, "gamma": 0.2},  # Highly regularized
        {"max_depth": 10, "min_child_weight": 1, "gamma": 0},  # Deep trees
        {
            "max_depth": 8,
            "min_child_weight": 2,
            "gamma": 0.05,
        },  # Balanced regularization
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
                    "depth": config["max_depth"],
                    "child_wt": config["min_child_weight"],
                    "gamma": f"{config['gamma']:.2f}",
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

    # Strategic regularization configurations
    configs = [
        {"reg_alpha": 0, "reg_lambda": 1},  # Default L2 only
        {"reg_alpha": 0, "reg_lambda": 0},  # No regularization
        {"reg_alpha": 1, "reg_lambda": 1},  # Balanced L1 + L2
        {"reg_alpha": 0.1, "reg_lambda": 1},  # Light L1 + L2
        {"reg_alpha": 1, "reg_lambda": 0},  # L1 only (sparsity)
        {"reg_alpha": 0, "reg_lambda": 10},  # Strong L2
        {"reg_alpha": 0.5, "reg_lambda": 0.5},  # Light balanced
        {"reg_alpha": 2, "reg_lambda": 2},  # Strong balanced
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
    """Optimize sampling configuration (subsample + colsample_bytree together)"""
    max_acc = -np.inf
    best_f1 = 0.0

    # Strategic sampling configurations
    configs = [
        {"subsample": 1.0, "colsample_bytree": 1.0},  # No sampling
        {"subsample": 0.8, "colsample_bytree": 0.8},  # Moderate sampling
        {"subsample": 0.9, "colsample_bytree": 0.9},  # Light sampling
        {"subsample": 0.7, "colsample_bytree": 0.7},  # Aggressive sampling
        {"subsample": 0.8, "colsample_bytree": 1.0},  # Row sampling only
        {"subsample": 1.0, "colsample_bytree": 0.8},  # Column sampling only
        {"subsample": 0.6, "colsample_bytree": 0.8},  # Heavy row sampling
        {"subsample": 0.8, "colsample_bytree": 0.6},  # Heavy column sampling
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
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts.update(best_config)
    return opts, max_acc, best_f1


def _optimize_objective_eval_metric(X_train, y_train, X_test, y_true, opts):
    """Optimize objective function and evaluation metric"""
    max_acc = -np.inf
    best_f1 = 0.0

    # Determine if binary or multi-class
    n_classes = len(np.unique(y_train))

    if n_classes == 2:  # Binary classification
        configs = [
            {"objective": "binary:logistic", "eval_metric": "logloss"},
            {"objective": "binary:logistic", "eval_metric": "error"},
            {"objective": "binary:logistic", "eval_metric": "auc"},
            {"objective": "binary:hinge", "eval_metric": "error"},
        ]
    else:  # Multi-class classification
        configs = [
            {"objective": "multi:softprob", "eval_metric": "mlogloss"},
            {"objective": "multi:softmax", "eval_metric": "merror"},
            {"objective": "multi:softprob", "eval_metric": "merror"},
        ]

    best_config = configs[0]

    with tqdm(configs, desc="Optimizing Objective/Metric", leave=False) as pbar:
        for config in pbar:
            test_opts = opts.copy()
            test_opts.update(config)

            # Add num_class for multi-class
            if n_classes > 2:
                test_opts["num_class"] = n_classes

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, **test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_config = config

            pbar.set_postfix(
                {
                    "objective": config["objective"][:10],
                    "metric": config["eval_metric"][:8],
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts.update(best_config)

    # Add num_class if multi-class
    if n_classes > 2:
        opts["num_class"] = n_classes

    return opts, max_acc, best_f1


def _optimize_advanced_params(X_train, y_train, X_test, y_true, opts):
    """Optimize advanced XGBoost parameters"""
    max_acc = -np.inf
    best_f1 = 0.0

    # Advanced parameter configurations
    configs = [
        {"scale_pos_weight": 1, "max_delta_step": 0},  # Default
        {"scale_pos_weight": "auto", "max_delta_step": 0},  # Auto-balance classes
        {"scale_pos_weight": 1, "max_delta_step": 1},  # Constrained updates
        {"scale_pos_weight": "auto", "max_delta_step": 1},  # Balanced + constrained
        {"scale_pos_weight": 1, "max_delta_step": 5},  # More constrained
        {
            "scale_pos_weight": "auto",
            "max_delta_step": 5,
        },  # Balanced + more constrained
    ]

    # Calculate actual scale_pos_weight for imbalanced datasets
    unique, counts = np.unique(y_train, return_counts=True)
    if len(unique) == 2:  # Binary classification
        neg_count = counts[0]
        pos_count = counts[1]
        auto_scale_pos_weight = neg_count / pos_count
    else:
        auto_scale_pos_weight = 1  # Not applicable for multi-class

    best_config = configs[0]

    with tqdm(configs, desc="Optimizing Advanced Params", leave=False) as pbar:
        for config in pbar:
            test_opts = opts.copy()

            # Handle auto scale_pos_weight
            if config["scale_pos_weight"] == "auto":
                test_opts["scale_pos_weight"] = auto_scale_pos_weight
            else:
                test_opts["scale_pos_weight"] = config["scale_pos_weight"]

            test_opts["max_delta_step"] = config["max_delta_step"]

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, **test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_config = config

            scale_str = (
                f"{auto_scale_pos_weight:.1f}"
                if config["scale_pos_weight"] == "auto"
                else str(config["scale_pos_weight"])
            )
            pbar.set_postfix(
                {
                    "scale_pos": scale_str,
                    "max_delta": config["max_delta_step"],
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    # Apply best config
    if best_config["scale_pos_weight"] == "auto":
        opts["scale_pos_weight"] = auto_scale_pos_weight
    else:
        opts["scale_pos_weight"] = best_config["scale_pos_weight"]

    opts["max_delta_step"] = best_config["max_delta_step"]

    return opts, max_acc, best_f1


def _optimize_xgboost(X_train, y_train, X_test, y_true, cycles=2):
    """
    Optimizes the hyperparameters for XGBoost Classifier.
    :param X_train: Training data
    :param y_train: Training labels
    :param X_test: Test data
    :param y_true: True labels for the test data
    :param cycles: Number of optimization cycles
    """

    if not XGBOOST_AVAILABLE:
        print("XGBoost is not available. Please install it with: pip install xgboost")
        return {}, 0.0, 0.0, [], []

    # Determine problem type
    n_classes = len(np.unique(y_train))

    # Define initial parameters with good defaults
    if n_classes == 2:  # Binary classification
        opts = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 6,
            "min_child_weight": 1,
            "gamma": 0,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_alpha": 0,
            "reg_lambda": 1,
            "scale_pos_weight": 1,
            "max_delta_step": 0,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
        }
    else:  # Multi-class classification
        opts = {
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "num_class": n_classes,
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 6,
            "min_child_weight": 1,
            "gamma": 0,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_alpha": 0,
            "reg_lambda": 1,
            "scale_pos_weight": 1,
            "max_delta_step": 0,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
        }

    # Track results
    ma_vec = []
    f1_vec = []

    # Main optimization loop with overall progress bar
    with tqdm(
        range(cycles), desc="XGBoost Optimization Cycles", position=0
    ) as cycle_pbar:
        for c in cycle_pbar:
            cycle_pbar.set_description(f"XGBoost Cycle {c + 1}/{cycles}")

            # Core hyperparameters (most impactful for XGBoost)
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
            opts, _, _ = _optimize_objective_eval_metric(
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
                    "depth": opts["max_depth"],
                    "reg_alpha": f"{opts['reg_alpha']:.1f}",
                    "reg_lambda": f"{opts['reg_lambda']:.1f}",
                    "subsample": f"{opts['subsample']:.1f}",
                }
            )

    return opts, ma, f1, ma_vec, f1_vec


def _analyze_xgboost_performance(X_train, y_train, X_test, y_true, best_opts):
    """Analyze XGBoost performance and feature importance"""

    if not XGBOOST_AVAILABLE:
        print("XGBoost is not available for analysis.")
        return {}

    print("\n" + "=" * 60)
    print("XGBOOST CLASSIFIER PERFORMANCE ANALYSIS")
    print("=" * 60)

    # Train final model
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)

    print("Training final XGBoost model with best parameters...")
    model = xgb.train(
        best_opts, dtrain, num_boost_round=best_opts["n_estimators"], verbose_eval=False
    )

    # Make predictions
    y_pred_proba = model.predict(dtest)

    # Convert probabilities to class predictions
    n_classes = len(np.unique(y_train))
    if n_classes == 2:  # Binary classification
        y_pred = (y_pred_proba > 0.5).astype(int)
    else:  # Multi-class classification
        y_pred = np.argmax(y_pred_proba.reshape(-1, n_classes), axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")

    print(f"\nFinal Model Performance:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 Score: {f1:.4f}")

    # Analyze configuration
    print(f"\nOptimal Configuration:")
    print(f"  Objective: {best_opts['objective']}")
    print(f"  Eval Metric: {best_opts['eval_metric']}")
    print(f"  Number of Estimators: {best_opts['n_estimators']}")
    print(f"  Learning Rate: {best_opts['learning_rate']:.3f}")
    print(f"  Max Depth: {best_opts['max_depth']}")
    print(f"  Min Child Weight: {best_opts['min_child_weight']}")
    print(f"  Gamma: {best_opts['gamma']:.3f}")
    print(f"  Subsample: {best_opts['subsample']:.2f}")
    print(f"  Column Sample by Tree: {best_opts['colsample_bytree']:.2f}")
    print(f"  Reg Alpha (L1): {best_opts['reg_alpha']:.2f}")
    print(f"  Reg Lambda (L2): {best_opts['reg_lambda']:.2f}")
    print(f"  Scale Pos Weight: {best_opts['scale_pos_weight']:.2f}")

    if best_opts["max_delta_step"] > 0:
        print(f"  Max Delta Step: {best_opts['max_delta_step']}")

    # Feature importance analysis
    print(f"\nFeature Importance Analysis:")
    importance_types = ["weight", "gain", "cover"]

    for imp_type in importance_types:
        try:
            importance = model.get_score(importance_type=imp_type)
            if importance:
                # Get top 5 features
                sorted_features = sorted(
                    importance.items(), key=lambda x: x[1], reverse=True
                )[:5]
                print(f"  Top 5 features by {imp_type}:")
                for feature, score in sorted_features:
                    print(f"    {feature}: {score:.4f}")
            else:
                print(f"  No {imp_type} importance available")
        except:
            print(f"  Could not compute {imp_type} importance")

    # Model characteristics
    print(f"\nModel Characteristics:")
    print(
        f"  Problem Type: {'Binary' if n_classes == 2 else 'Multi-class'} Classification"
    )
    print(f"  Number of Classes: {n_classes}")
    print(f"  Training Samples: {X_train.shape[0]}")
    print(f"  Features: {X_train.shape[1]}")

    # Check for class imbalance
    unique, counts = np.unique(y_train, return_counts=True)
    class_distribution = dict(zip(unique, counts))
    print(f"  Class Distribution: {class_distribution}")

    # Imbalance ratio
    if len(unique) == 2:
        imbalance_ratio = max(counts) / min(counts)
        print(f"  Imbalance Ratio: {imbalance_ratio:.2f}:1")
        if imbalance_ratio > 2:
            print(
                f"  Dataset is imbalanced - scale_pos_weight optimization was important"
            )

    # Regularization analysis
    total_reg = best_opts["reg_alpha"] + best_opts["reg_lambda"]
    if total_reg == 0:
        print(f"  Regularization: None (may overfit)")
    elif total_reg < 1:
        print(f"  Regularization: Light")
    elif total_reg < 5:
        print(f"  Regularization: Moderate")
    else:
        print(f"  Regularization: Strong")

    # Sampling analysis
    total_sampling = best_opts["subsample"] * best_opts["colsample_bytree"]
    if total_sampling >= 0.9:
        print(f"  Sampling: Minimal (uses most data)")
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
    }


# Example usage function
def example_usage():
    """Example of how to use the optimized XGBoost Classifier function"""
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    if not XGBOOST_AVAILABLE:
        print("XGBoost is not available. Please install it with: pip install xgboost")
        return None, 0.0, 0.0, [], []

    # Generate sample data
    print("Generating sample classification data...")
    X, y = make_classification(
        n_samples=3000,  # Larger dataset for XGBoost
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

    # XGBoost can handle raw features well, but scaling can help
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Training set: {X_train_scaled.shape}")
    print(f"Test set: {X_test_scaled.shape}")
    print("Starting XGBoost Classifier optimization...")

    # Run optimization
    best_opts, best_acc, best_f1, acc_history, f1_history = (
        _optimize_xgboost_classifier(
            X_train_scaled, y_train, X_test_scaled, y_test, cycles=2
        )
    )

    print(f"\nOptimization completed!")
    print(f"Best accuracy: {best_acc:.4f}")
    print(f"Best F1 score: {best_f1:.4f}")

    # Analyze performance
    analysis = _analyze_xgboost_performance(
        X_train_scaled, y_train, X_test_scaled, y_test, best_opts
    )

    return best_opts, best_acc, best_f1, acc_history, f1_history
