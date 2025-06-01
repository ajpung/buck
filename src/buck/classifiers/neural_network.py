from typing import Any
import warnings
import numpy as np
import time
from tqdm.auto import tqdm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.exceptions import ConvergenceWarning

# Suppress warnings for cleaner progress bars
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Global efficiency controls
_max_time_per_step = (
    900  # 10 minutes max per optimization step (neural nets can be slow)
)
_max_time_per_model = 900  # 3 minutes max per model evaluation
_min_accuracy_threshold = 0.15  # Stop if accuracy is consistently terrible
_consecutive_failures = 0
_max_consecutive_failures = 3


def _safe_evaluate_model(X_train, y_train, X_test, y_true, **kwargs):
    """Safely evaluate a neural network model configuration with timeout protection"""
    global _consecutive_failures

    try:
        start_time = time.time()
        classifier = MLPClassifier(**kwargs)
        classifier.fit(X_train, y_train)

        # Check if training took too long
        # training_time = time.time() - start_time
        # if training_time > _max_time_per_model:
        #    print(f"⏰ Neural net timeout after {training_time:.1f}s")
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


def _get_smart_architectures(n_samples, n_features):
    """Get smart neural network architectures based on dataset characteristics"""

    # Rule of thumb: hidden layer size between n_features and n_classes
    # For classification, good starting point is 2/3 of input + output size

    architectures = []

    # Single layer architectures (often work well)
    if n_features <= 20:
        single_layers = [(50,), (100,), (n_features * 2,), (n_features * 4,)]
    elif n_features <= 100:
        single_layers = [(100,), (200,), (n_features,), (n_features * 2,)]
    else:
        single_layers = [(100,), (200,), (300,), (n_features // 2,)]

    architectures.extend(single_layers)

    # Two layer architectures (for complex patterns)
    if n_samples > 500:  # Only for sufficient data
        if n_features <= 50:
            two_layers = [(100, 50), (200, 100), (n_features * 2, n_features)]
        else:
            two_layers = [(200, 100), (300, 150), (150, 75)]

        architectures.extend(two_layers)

    # Three layer architectures (only for large datasets)
    if n_samples > 2000:
        three_layers = [(200, 100, 50), (300, 150, 75)]
        architectures.extend(three_layers)

    return architectures


def _optimize_architecture_and_solver(X_train, y_train, X_test, y_true, opts):
    """Optimize architecture and solver together for efficiency"""
    global _consecutive_failures
    start_time = time.time()
    _consecutive_failures = 0

    max_acc = -np.inf
    best_f1 = 0.0
    n_samples, n_features = X_train.shape

    # Get smart architectures
    architectures = _get_smart_architectures(n_samples, n_features)

    # Smart solver selection based on dataset size
    if n_samples < 1000:
        solvers = ["lbfgs", "adam"]  # lbfgs is good for small datasets
    elif n_samples < 10000:
        solvers = ["adam", "lbfgs"]  # adam is usually best
    else:
        solvers = ["adam"]  # adam scales best to large datasets

    # Create architecture-solver combinations
    configs = []
    for arch in architectures:
        for solver in solvers:
            configs.append({"hidden_layer_sizes": arch, "solver": solver})

    best_config = configs[0]

    with tqdm(configs, desc="Optimizing Architecture & Solver", leave=False) as pbar:
        for config in pbar:
            # Early stopping conditions
            if _consecutive_failures >= _max_consecutive_failures:
                pbar.set_description("Architecture & Solver (POOR ACCURACY)")
                break

            test_opts = opts.copy()
            test_opts.update(config)

            # Adjust max_iter based on solver
            if config["solver"] == "lbfgs":
                test_opts["max_iter"] = min(test_opts["max_iter"], 1000)

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, **test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_config = config

            arch_str = str(config["hidden_layer_sizes"])[:10]
            pbar.set_postfix(
                {
                    "arch": arch_str,
                    "solver": config["solver"][:4],
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts.update(best_config)
    return opts, max_acc, best_f1


def _optimize_activation(X_train, y_train, X_test, y_true, opts):
    """Optimize activation function with early stopping"""
    global _consecutive_failures
    start_time = time.time()
    _consecutive_failures = 0

    max_acc = -np.inf
    best_f1 = 0.0

    # Most effective activation functions
    variable_array = ["relu", "tanh", "logistic"]
    best_val = "relu"  # Good default

    with tqdm(variable_array, desc="Optimizing Activation", leave=False) as pbar:
        for v in pbar:
            if _consecutive_failures >= _max_consecutive_failures:
                pbar.set_description("Activation (POOR ACCURACY)")
                break

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


def _optimize_regularization(X_train, y_train, X_test, y_true, opts):
    """Optimize regularization (alpha) with smart range"""
    global _consecutive_failures
    start_time = time.time()
    _consecutive_failures = 0

    max_acc = -np.inf
    best_f1 = 0.0
    n_samples = X_train.shape[0]

    # Smart alpha range based on dataset size
    if n_samples < 1000:
        # Smaller datasets need more regularization
        variable_array = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    elif n_samples < 10000:
        # Medium datasets - balanced range
        variable_array = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    else:
        # Large datasets can handle less regularization
        variable_array = [1e-6, 1e-5, 1e-4, 1e-3]

    best_val = 1e-4  # sklearn default

    with tqdm(variable_array, desc="Optimizing Regularization", leave=False) as pbar:
        for v in pbar:
            # Early stopping conditions
            if _consecutive_failures >= _max_consecutive_failures:
                pbar.set_description("Regularization (POOR ACCURACY)")
                break

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


def _optimize_learning_params(X_train, y_train, X_test, y_true, opts):
    """Optimize learning rate and related parameters together"""
    global _consecutive_failures
    start_time = time.time()
    _consecutive_failures = 0

    max_acc = -np.inf
    best_f1 = 0.0

    # Learning rate configurations based on solver
    if opts["solver"] == "adam":
        configs = [
            {"learning_rate_init": 0.001},  # Default
            {"learning_rate_init": 0.01},
            {"learning_rate_init": 0.0001},
            {"learning_rate_init": 0.005},
        ]
    elif opts["solver"] == "sgd":
        configs = [
            {"learning_rate_init": 0.1, "learning_rate": "constant"},
            {"learning_rate_init": 0.01, "learning_rate": "constant"},
            {"learning_rate_init": 0.1, "learning_rate": "adaptive"},
            {"learning_rate_init": 0.01, "learning_rate": "adaptive"},
        ]
    else:  # lbfgs doesn't use learning rate
        return opts, max_acc, best_f1

    best_config = configs[0]

    with tqdm(configs, desc="Optimizing Learning Params", leave=False) as pbar:
        for config in pbar:
            # Early stopping conditions
            if _consecutive_failures >= _max_consecutive_failures:
                pbar.set_description("Learning Params (POOR ACCURACY)")
                break

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
                    "lr_init": f"{config['learning_rate_init']:.4f}",
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts.update(best_config)
    return opts, max_acc, best_f1


def _optimize_convergence_params(X_train, y_train, X_test, y_true, opts):
    """Optimize convergence parameters together"""
    global _consecutive_failures
    start_time = time.time()
    _consecutive_failures = 0

    max_acc = -np.inf
    best_f1 = 0.0
    n_samples = X_train.shape[0]

    # Smart convergence parameters based on dataset size and solver
    if opts["solver"] == "lbfgs":
        configs = [
            {"max_iter": 500, "tol": 1e-4, "early_stopping": False},
            {"max_iter": 1000, "tol": 1e-4, "early_stopping": False},
            {"max_iter": 1000, "tol": 1e-5, "early_stopping": False},
        ]
    else:  # adam or sgd
        if n_samples < 1000:
            configs = [
                {"max_iter": 500, "tol": 1e-4, "early_stopping": False},
                {
                    "max_iter": 1000,
                    "tol": 1e-4,
                    "early_stopping": True,
                    "validation_fraction": 0.1,
                },
                {
                    "max_iter": 2000,
                    "tol": 1e-4,
                    "early_stopping": True,
                    "validation_fraction": 0.1,
                },
            ]
        else:
            configs = [
                {
                    "max_iter": 1000,
                    "tol": 1e-4,
                    "early_stopping": True,
                    "validation_fraction": 0.1,
                },
                {
                    "max_iter": 2000,
                    "tol": 1e-4,
                    "early_stopping": True,
                    "validation_fraction": 0.1,
                },
                {
                    "max_iter": 2000,
                    "tol": 1e-5,
                    "early_stopping": True,
                    "validation_fraction": 0.1,
                },
            ]

    best_config = configs[0]

    with tqdm(configs, desc="Optimizing Convergence", leave=False) as pbar:
        for config in pbar:
            # Early stopping conditions
            if _consecutive_failures >= _max_consecutive_failures:
                pbar.set_description("Convergence (POOR ACCURACY)")
                break

            test_opts = opts.copy()
            test_opts.update(config)

            # Add early stopping parameters if enabled
            if config.get("early_stopping", False):
                test_opts["n_iter_no_change"] = 10

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
                    "early_stop": "Y" if config.get("early_stopping", False) else "N",
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts.update(best_config)
    return opts, max_acc, best_f1


def _optimize_neural_network(X_train, y_train, X_test, y_true, cycles=2):
    """
    FAST optimized hyperparameters for MLPClassifier (Neural Network).
    :param X_train: Training data
    :param y_train: Training labels
    :param X_test: Test data
    :param y_true: True labels for the test data
    :param cycles: Number of optimization cycles
    """

    n_samples, n_features = X_train.shape
    n_classes = len(np.unique(y_train))
    print(f"Dataset: {n_samples} samples, {n_features} features, {n_classes} classes")

    # Neural network warnings
    if n_samples < 100:
        print("⚠️  WARNING: Very small dataset for neural networks!")
        print("Neural networks typically need hundreds or thousands of samples.")
        print(
            "Consider simpler algorithms (LogisticRegression, RandomForest) for small data."
        )

    if n_features > n_samples:
        print("⚠️  WARNING: More features than samples!")
        print(
            "Neural networks may overfit. Consider dimensionality reduction or regularization."
        )

    # Quick baseline check
    print("Running baseline check...")
    baseline_nn = MLPClassifier(
        hidden_layer_sizes=(100,), random_state=42, max_iter=500, early_stopping=True
    )
    start_time = time.time()
    baseline_nn.fit(X_train, y_train)
    baseline_time = time.time() - start_time
    baseline_acc = baseline_nn.score(X_test, y_true)

    print(f"Baseline Neural Net: {baseline_acc:.4f} (trained in {baseline_time:.1f}s)")

    if baseline_time > 30:
        print(
            "⚠️  Slow baseline training - neural net will be time-consuming on this dataset"
        )

    if baseline_acc < 0.2:
        print("⚠️  WARNING: Very low baseline accuracy!")
        print("Consider feature scaling, different architectures, or other algorithms.")

    # Define initial parameters with smart defaults
    opts = {
        "hidden_layer_sizes": (100,),
        "activation": "relu",
        "solver": "adam",
        "alpha": 1e-4,
        "batch_size": "auto",
        "learning_rate": "constant",
        "learning_rate_init": 0.001,
        "max_iter": 1000,
        "shuffle": True,
        "random_state": 42,  # Fixed for reproducibility (no need to optimize)
        "tol": 1e-4,
        "verbose": False,
        "warm_start": False,
        "momentum": 0.9,
        "nesterovs_momentum": True,
        "early_stopping": True,  # Enable by default for speed
        "validation_fraction": 0.1,
        "beta_1": 0.9,
        "beta_2": 0.999,
        "epsilon": 1e-8,
        "n_iter_no_change": 10,
        "max_fun": 15000,
    }

    # Track results
    ma_vec = []
    f1_vec = []

    # Main optimization loop
    with tqdm(
        range(cycles), desc="FAST Neural Network Optimization", position=0
    ) as cycle_pbar:
        for c in cycle_pbar:
            cycle_start_time = time.time()
            cycle_pbar.set_description(f"Neural Net Cycle {c + 1}/{cycles}")

            # Core optimizations in order of importance
            opts, _, _ = _optimize_architecture_and_solver(
                X_train, y_train, X_test, y_true, opts
            )
            opts, _, _ = _optimize_activation(X_train, y_train, X_test, y_true, opts)
            opts, _, _ = _optimize_regularization(
                X_train, y_train, X_test, y_true, opts
            )
            opts, _, _ = _optimize_learning_params(
                X_train, y_train, X_test, y_true, opts
            )
            opts, ma, f1 = _optimize_convergence_params(
                X_train, y_train, X_test, y_true, opts
            )

            ma_vec.append(ma)
            f1_vec.append(f1)

            cycle_time = time.time() - cycle_start_time

            cycle_pbar.set_postfix(
                {
                    "accuracy": f"{ma:.4f}",
                    "f1": f"{f1:.4f}",
                    "best_overall": f"{max(ma_vec):.4f}",
                    "cycle_time": f"{cycle_time:.1f}s",
                    "arch": str(opts["hidden_layer_sizes"])[:12],
                    "solver": opts["solver"],
                    "alpha": f'{opts["alpha"]:.0e}',
                    "lr": f'{opts.get("learning_rate_init", 0):.4f}',
                    "baseline_beat": f"{ma - baseline_acc:+.4f}",
                }
            )

    return opts, ma, f1, ma_vec, f1_vec
