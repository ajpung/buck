from typing import Any
import warnings
import numpy as np
from tqdm.auto import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.exceptions import ConvergenceWarning

# Suppress convergence warnings for cleaner progress bars
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def _safe_evaluate_model(X_train, y_train, X_test, y_true, **kwargs):
    """Safely evaluate a model configuration"""
    try:
        classifier = RandomForestClassifier(**kwargs)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        return accuracy, f1, True
    except Exception:
        return 0.0, 0.0, False


def _optimize_rs(X_train, y_train, X_test, y_true, opts):
    """Optimize random state"""
    max_acc = -np.inf
    best_f1 = 0.0
    variable_array = np.arange(30)  # Reduced from 800 to 30
    best_val = variable_array[0]

    with tqdm(variable_array, desc="Optimizing Random State", leave=False) as pbar:
        for v in pbar:
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
    max_acc = -np.inf
    best_f1 = 0.0
    # Strategic values for Random Forest estimators
    variable_array = [10, 25, 50, 100, 200, 300, 500, 800]
    best_val = variable_array[0]

    with tqdm(variable_array, desc="Optimizing N Estimators", leave=False) as pbar:
        for v in pbar:
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
    max_acc = -np.inf
    best_f1 = 0.0
    # Strategic depths for Random Forest
    variable_array = [3, 5, 7, 10, 15, 20, 25, None]
    best_val = variable_array[0]

    with tqdm(variable_array, desc="Optimizing Max Depth", leave=False) as pbar:
        for v in pbar:
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
    max_acc = -np.inf
    best_f1 = 0.0
    n_features = X_train.shape[1]

    # Intelligent feature selection based on dataset size
    if n_features <= 10:
        feature_options = [1, 2, 3, "sqrt", "log2", None]
    elif n_features <= 50:
        feature_options = [5, 10, 20, "sqrt", "log2", None]
    else:
        feature_options = [10, 20, 50, n_features // 4, "sqrt", "log2", None]

    # Remove duplicates and invalid options
    variable_array = []
    for opt in feature_options:
        if isinstance(opt, int) and opt <= n_features and opt not in variable_array:
            variable_array.append(opt)
        elif isinstance(opt, str) and opt not in variable_array:
            variable_array.append(opt)
        elif opt is None and opt not in variable_array:
            variable_array.append(opt)

    best_val = variable_array[0]

    with tqdm(variable_array, desc="Optimizing Max Features", leave=False) as pbar:
        for v in pbar:
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
    max_acc = -np.inf
    best_f1 = 0.0
    variable_array = ["gini", "entropy", "log_loss"]
    best_val = variable_array[0]

    with tqdm(variable_array, desc="Optimizing Criterion", leave=False) as pbar:
        for v in pbar:
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
    max_acc = -np.inf
    best_f1 = 0.0

    # Combined configurations for efficiency
    configs = [
        {"min_samples_split": 2, "min_samples_leaf": 1},
        {"min_samples_split": 5, "min_samples_leaf": 1},
        {"min_samples_split": 10, "min_samples_leaf": 1},
        {"min_samples_split": 2, "min_samples_leaf": 2},
        {"min_samples_split": 5, "min_samples_leaf": 2},
        {"min_samples_split": 10, "min_samples_leaf": 2},
        {"min_samples_split": 20, "min_samples_leaf": 5},
        {"min_samples_split": 50, "min_samples_leaf": 10},
    ]
    best_config = configs[0]

    with tqdm(configs, desc="Optimizing Min Samples Config", leave=False) as pbar:
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
    max_acc = -np.inf
    best_f1 = 0.0

    # Combined regularization configurations
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
        {
            "min_weight_fraction_leaf": 0.05,
            "min_impurity_decrease": 0.0,
            "ccp_alpha": 0.0,
        },
        {
            "min_weight_fraction_leaf": 0.0,
            "min_impurity_decrease": 0.05,
            "ccp_alpha": 0.0,
        },
        {
            "min_weight_fraction_leaf": 0.0,
            "min_impurity_decrease": 0.0,
            "ccp_alpha": 0.05,
        },
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
    max_acc = -np.inf
    best_f1 = 0.0

    # Bootstrap configurations
    configs = [
        {"bootstrap": True, "oob_score": False, "max_samples": None},
        {"bootstrap": True, "oob_score": True, "max_samples": None},
        {"bootstrap": True, "oob_score": False, "max_samples": 0.8},
        {"bootstrap": True, "oob_score": True, "max_samples": 0.8},
        {"bootstrap": False, "oob_score": False, "max_samples": None},
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
    max_acc = -np.inf
    best_f1 = 0.0
    variable_array = [None, "balanced", "balanced_subsample"]
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


def _optimize_max_leaf_nodes(X_train, y_train, X_test, y_true, opts):
    """Optimize max leaf nodes"""
    max_acc = -np.inf
    best_f1 = 0.0
    # Strategic leaf node values
    variable_array = [10, 50, 100, 200, 500, 1000, None]
    best_val = variable_array[0]

    with tqdm(variable_array, desc="Optimizing Max Leaf Nodes", leave=False) as pbar:
        for v in pbar:
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

    # Define initial parameters with better defaults
    opts = {
        "n_estimators": 100,
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

            cycle_pbar.set_postfix(
                {
                    "accuracy": f"{ma:.4f}",
                    "f1": f"{f1:.4f}",
                    "best_overall": f"{max(ma_vec):.4f}",
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
