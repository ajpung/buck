from typing import Any
import warnings
import numpy as np
from tqdm.auto import tqdm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.exceptions import ConvergenceWarning

# Suppress warnings for cleaner progress bars
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def _safe_evaluate_model(X_train, y_train, X_test, y_true, **kwargs):
    """Safely evaluate a model configuration"""
    try:
        classifier = MLPClassifier(**kwargs)
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
    variable_array = np.arange(30)  # Reduced from 150
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


def _optimize_hidden_layers(X_train, y_train, X_test, y_true, opts):
    """Optimize hidden layer architecture"""
    max_acc = -np.inf
    best_f1 = 0.0

    # More intelligent architectures based on input size
    n_features = X_train.shape[1]

    # Single layer architectures
    single_layers = [
        (n_features // 4,),
        (n_features // 2,),
        (n_features,),
        (n_features * 2,),
        (50,),
        (100,),
        (200,),
        (300,),
    ]

    # Two layer architectures
    two_layers = [
        (100, 50),
        (200, 100),
        (300, 150),
        (150, 75),
        (n_features, n_features // 2),
        (n_features // 2, n_features // 4),
    ]

    # Three layer architectures
    three_layers = [(200, 100, 50), (300, 150, 75), (150, 100, 50)]

    variable_array = single_layers + two_layers + three_layers
    best_val = variable_array[0]

    with tqdm(variable_array, desc="Optimizing Hidden Layers", leave=False) as pbar:
        for v in pbar:
            test_opts = opts.copy()
            test_opts["hidden_layer_sizes"] = v

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, **test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_val = v

            pbar.set_postfix(
                {
                    "layers": str(v)[:15],
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts["hidden_layer_sizes"] = best_val
    return opts, max_acc, best_f1


def _optimize_activation(X_train, y_train, X_test, y_true, opts):
    """Optimize activation function"""
    max_acc = -np.inf
    best_f1 = 0.0
    variable_array = ["relu", "tanh", "logistic"]  # Removed 'identity' (rarely useful)
    best_val = variable_array[0]

    with tqdm(variable_array, desc="Optimizing Activation", leave=False) as pbar:
        for v in pbar:
            test_opts = opts.copy()
            test_opts["activation"] = v

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, **test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_val = v

            pbar.set_postfix(
                {
                    "activation": v,
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts["activation"] = best_val
    return opts, max_acc, best_f1


def _optimize_solver(X_train, y_train, X_test, y_true, opts):
    """Optimize solver"""
    max_acc = -np.inf
    best_f1 = 0.0
    variable_array = ["adam", "lbfgs", "sgd"]
    best_val = variable_array[0]

    with tqdm(variable_array, desc="Optimizing Solver", leave=False) as pbar:
        for v in pbar:
            test_opts = opts.copy()
            test_opts["solver"] = v

            # Adjust max_iter based on solver
            if v == "lbfgs":
                test_opts["max_iter"] = min(
                    test_opts["max_iter"], 1000
                )  # lbfgs converges faster

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, **test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_val = v

            pbar.set_postfix(
                {
                    "solver": v,
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts["solver"] = best_val
    return opts, max_acc, best_f1


def _optimize_alpha(X_train, y_train, X_test, y_true, opts):
    """Optimize alpha (L2 regularization)"""
    max_acc = -np.inf
    best_f1 = 0.0
    variable_array = np.logspace(-6, -1, 12)  # Much more efficient: 1e-6 to 1e-1
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


def _optimize_learning_rate_init(X_train, y_train, X_test, y_true, opts):
    """Optimize initial learning rate"""
    max_acc = -np.inf
    best_f1 = 0.0
    variable_array = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    best_val = variable_array[0]

    with tqdm(variable_array, desc="Optimizing Learning Rate", leave=False) as pbar:
        for v in pbar:
            test_opts = opts.copy()
            test_opts["learning_rate_init"] = v

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, **test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_val = v

            pbar.set_postfix(
                {
                    "lr_init": f"{v:.4f}",
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts["learning_rate_init"] = best_val
    return opts, max_acc, best_f1


def _optimize_batch_size(X_train, y_train, X_test, y_true, opts):
    """Optimize batch size"""
    max_acc = -np.inf
    best_f1 = 0.0

    # Skip batch size optimization for lbfgs (doesn't use batches)
    if opts.get("solver") == "lbfgs":
        return opts, max_acc, best_f1

    n_samples = X_train.shape[0]
    variable_array = ["auto", 32, 64, 128, 256, min(512, n_samples // 4)]
    best_val = variable_array[0]

    with tqdm(variable_array, desc="Optimizing Batch Size", leave=False) as pbar:
        for v in pbar:
            test_opts = opts.copy()
            test_opts["batch_size"] = v

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, **test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_val = v

            pbar.set_postfix(
                {
                    "batch": str(v),
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts["batch_size"] = best_val
    return opts, max_acc, best_f1


def _optimize_learning_rate_schedule(X_train, y_train, X_test, y_true, opts):
    """Optimize learning rate schedule"""
    max_acc = -np.inf
    best_f1 = 0.0

    # Only for SGD solver
    if opts.get("solver") != "sgd":
        return opts, max_acc, best_f1

    variable_array = ["constant", "adaptive", "invscaling"]
    best_val = variable_array[0]

    with tqdm(variable_array, desc="Optimizing LR Schedule", leave=False) as pbar:
        for v in pbar:
            test_opts = opts.copy()
            test_opts["learning_rate"] = v

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, **test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_val = v

            pbar.set_postfix(
                {
                    "lr_sched": v[:8],
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts["learning_rate"] = best_val
    return opts, max_acc, best_f1


def _optimize_momentum(X_train, y_train, X_test, y_true, opts):
    """Optimize momentum (SGD only)"""
    max_acc = -np.inf
    best_f1 = 0.0

    # Only for SGD solver
    if opts.get("solver") != "sgd":
        return opts, max_acc, best_f1

    variable_array = [0.5, 0.7, 0.9, 0.95, 0.99]
    best_val = variable_array[0]

    with tqdm(variable_array, desc="Optimizing Momentum", leave=False) as pbar:
        for v in pbar:
            test_opts = opts.copy()
            test_opts["momentum"] = v

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, **test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_val = v

            pbar.set_postfix(
                {
                    "momentum": f"{v:.2f}",
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts["momentum"] = best_val
    return opts, max_acc, best_f1


def _optimize_adam_params(X_train, y_train, X_test, y_true, opts):
    """Optimize Adam-specific parameters"""
    max_acc = -np.inf
    best_f1 = 0.0

    # Only for Adam solver
    if opts.get("solver") != "adam":
        return opts, max_acc, best_f1

    # Optimize beta_1 and beta_2 together for efficiency
    beta_combinations = [
        (0.9, 0.999),  # Default
        (0.8, 0.999),
        (0.95, 0.999),
        (0.9, 0.99),
        (0.9, 0.9999),
        (0.85, 0.995),
    ]

    best_val = beta_combinations[0]

    with tqdm(beta_combinations, desc="Optimizing Adam Betas", leave=False) as pbar:
        for beta1, beta2 in pbar:
            test_opts = opts.copy()
            test_opts["beta_1"] = beta1
            test_opts["beta_2"] = beta2

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, **test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_val = (beta1, beta2)

            pbar.set_postfix(
                {
                    "beta1": f"{beta1:.2f}",
                    "beta2": f"{beta2:.3f}",
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts["beta_1"], opts["beta_2"] = best_val
    return opts, max_acc, best_f1


def _optimize_early_stopping(X_train, y_train, X_test, y_true, opts):
    """Optimize early stopping"""
    max_acc = -np.inf
    best_f1 = 0.0
    variable_array = [True, False]
    best_val = variable_array[0]

    with tqdm(variable_array, desc="Optimizing Early Stopping", leave=False) as pbar:
        for v in pbar:
            test_opts = opts.copy()
            test_opts["early_stopping"] = v

            if v:  # If early stopping is enabled
                test_opts["validation_fraction"] = 0.1
                test_opts["n_iter_no_change"] = 10

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, **test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_val = v

            pbar.set_postfix(
                {
                    "early_stop": str(v),
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts["early_stopping"] = best_val
    return opts, max_acc, best_f1


def _optimize_max_iter(X_train, y_train, X_test, y_true, opts):
    """Optimize maximum iterations"""
    max_acc = -np.inf
    best_f1 = 0.0

    # Different ranges based on solver
    if opts.get("solver") == "lbfgs":
        variable_array = [200, 500, 1000, 2000]
    else:
        variable_array = [200, 500, 1000, 2000, 5000]

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


def _optimize_tolerance(X_train, y_train, X_test, y_true, opts):
    """Optimize tolerance"""
    max_acc = -np.inf
    best_f1 = 0.0
    variable_array = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    best_val = variable_array[0]

    with tqdm(variable_array, desc="Optimizing Tolerance", leave=False) as pbar:
        for v in pbar:
            test_opts = opts.copy()
            test_opts["tol"] = v

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, **test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_val = v

            pbar.set_postfix(
                {
                    "tol": f"{v:.0e}",
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts["tol"] = best_val
    return opts, max_acc, best_f1


def _optimize_neural_network(X_train, y_train, X_test, y_true, cycles=2):
    """
    Optimizes the hyperparameters for MLPClassifier (Neural Network).
    :param X_train: Training data
    :param y_train: Training labels
    :param X_test: Test data
    :param y_true: True labels for the test data
    :param cycles: Number of optimization cycles
    """

    # Define initial parameters
    opts = {
        "hidden_layer_sizes": (100,),
        "activation": "relu",
        "solver": "adam",
        "alpha": 1e-4,
        "batch_size": "auto",
        "learning_rate": "constant",
        "learning_rate_init": 0.001,
        "power_t": 0.5,
        "max_iter": 1000,
        "shuffle": True,
        "random_state": 42,
        "tol": 1e-4,
        "verbose": False,
        "warm_start": False,
        "momentum": 0.9,
        "nesterovs_momentum": True,
        "early_stopping": False,
        "validation_fraction": 0.1,
        "beta_1": 0.9,
        "beta_2": 0.999,
        "epsilon": 1e-8,
        "n_iter_no_change": 10,
        "max_fun": 15000,
    }

    # Optimize hyperparameters
    ma_vec = []
    f1_vec = []

    # Main optimization loop with overall progress bar
    with tqdm(range(cycles), desc="Optimization Cycles", position=0) as cycle_pbar:
        for c in cycle_pbar:
            cycle_pbar.set_description(f"Neural Network Cycle {c + 1}/{cycles}")

            # Core architecture and algorithm optimization
            opts, _, _ = _optimize_rs(X_train, y_train, X_test, y_true, opts)
            opts, _, _ = _optimize_hidden_layers(X_train, y_train, X_test, y_true, opts)
            opts, _, _ = _optimize_activation(X_train, y_train, X_test, y_true, opts)
            opts, _, _ = _optimize_solver(X_train, y_train, X_test, y_true, opts)

            # Regularization and convergence
            opts, _, _ = _optimize_alpha(X_train, y_train, X_test, y_true, opts)
            opts, _, _ = _optimize_tolerance(X_train, y_train, X_test, y_true, opts)
            opts, _, _ = _optimize_max_iter(X_train, y_train, X_test, y_true, opts)
            opts, _, _ = _optimize_early_stopping(
                X_train, y_train, X_test, y_true, opts
            )

            # Solver-specific optimizations
            opts, _, _ = _optimize_learning_rate_init(
                X_train, y_train, X_test, y_true, opts
            )
            opts, _, _ = _optimize_batch_size(X_train, y_train, X_test, y_true, opts)
            opts, _, _ = _optimize_learning_rate_schedule(
                X_train, y_train, X_test, y_true, opts
            )
            opts, _, _ = _optimize_momentum(X_train, y_train, X_test, y_true, opts)
            opts, ma, f1 = _optimize_adam_params(X_train, y_train, X_test, y_true, opts)

            ma_vec.append(ma)
            f1_vec.append(f1)

            cycle_pbar.set_postfix(
                {
                    "accuracy": f"{ma:.4f}",
                    "f1": f"{f1:.4f}",
                    "best_overall": f"{max(ma_vec):.4f}",
                    "layers": str(opts["hidden_layer_sizes"])[:15],
                    "solver": opts["solver"],
                    "alpha": f'{opts["alpha"]:.0e}',
                    "lr": f'{opts["learning_rate_init"]:.4f}',
                }
            )

    return opts, ma, f1, ma_vec, f1_vec
