from typing import Any
import warnings
import numpy as np
from tqdm.auto import tqdm
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import (
    RBF,
    Matern,
    RationalQuadratic,
    ExpSineSquared,
    DotProduct,
    WhiteKernel,
    ConstantKernel,
)
from sklearn.metrics import accuracy_score, f1_score
from sklearn.exceptions import ConvergenceWarning

# Suppress convergence warnings for cleaner progress bars
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def _safe_evaluate_model(X_train, y_train, X_test, y_true, opts):
    """Safely evaluate a model configuration"""
    try:
        classifier = GaussianProcessClassifier(**opts)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        return accuracy, f1, True
    except Exception:
        return 0.0, 0.0, False


def _optimize_kernel_type(X_train, y_train, X_test, y_true, opts):
    """Optimize kernel type"""
    max_acc = -np.inf
    best_f1 = 0.0

    # Define different kernel options
    kernels = {
        "rbf": 1.0 * RBF(1.0),
        "matern_1.5": 1.0 * Matern(length_scale=1.0, nu=1.5),
        "matern_2.5": 1.0 * Matern(length_scale=1.0, nu=2.5),
        "rational_quadratic": 1.0 * RationalQuadratic(length_scale=1.0, alpha=1.0),
        "dot_product": DotProduct(sigma_0=1.0),
        "rbf_white": 1.0 * RBF(1.0) + WhiteKernel(noise_level=1e-5),
        "matern_white": 1.0 * Matern(length_scale=1.0, nu=1.5)
        + WhiteKernel(noise_level=1e-5),
    }

    best_kernel = "rbf"
    kernel_items = list(kernels.items())

    with tqdm(kernel_items, desc="Optimizing Kernel Type", leave=False) as pbar:
        for kernel_name, kernel in pbar:
            test_opts = opts.copy()
            test_opts["kernel"] = kernel

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_kernel = kernel_name
                opts["kernel"] = kernel

            pbar.set_postfix(
                {
                    "current": kernel_name[:8],
                    "best_acc": f"{max_acc:.4f}",
                    "best": best_kernel[:8],
                }
            )

    return opts, max_acc, best_f1


def _optimize_kernel_length_scale(X_train, y_train, X_test, y_true, opts):
    """Optimize kernel length scale"""
    max_acc = -np.inf
    best_f1 = 0.0
    variable_array = np.logspace(-2, 2, 10)  # More granular search
    current_kernel = opts["kernel"]

    with tqdm(
        variable_array, desc="Optimizing Kernel Length Scale", leave=False
    ) as pbar:
        for length_scale in pbar:
            new_kernel = _create_kernel_with_length_scale(current_kernel, length_scale)

            if new_kernel is None:
                pbar.set_postfix({"scale": f"{length_scale:.3f}", "status": "skipped"})
                continue

            test_opts = opts.copy()
            test_opts["kernel"] = new_kernel

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                opts["kernel"] = new_kernel

            pbar.set_postfix(
                {
                    "scale": f"{length_scale:.3f}",
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best": f"{max_acc:.4f}",
                }
            )

    return opts, max_acc, best_f1


def _create_kernel_with_length_scale(kernel, length_scale):
    """Create new kernel with updated length scale"""
    try:
        # Handle composite kernels (with WhiteKernel)
        if hasattr(kernel, "k1") and hasattr(kernel, "k2"):
            # Find the base kernel and noise kernel
            if isinstance(kernel.k1, (RBF, Matern, RationalQuadratic)):
                base_kernel = kernel.k1
                noise_kernel = kernel.k2
            elif isinstance(kernel.k2, (RBF, Matern, RationalQuadratic)):
                base_kernel = kernel.k2
                noise_kernel = kernel.k1
            else:
                return None

            new_base = _create_simple_kernel_with_scale(base_kernel, length_scale)
            return new_base + noise_kernel if new_base else None

        # Handle simple kernels
        return _create_simple_kernel_with_scale(kernel, length_scale)

    except Exception:
        return None


def _create_simple_kernel_with_scale(kernel, length_scale):
    """Create simple kernel with specified length scale"""
    if isinstance(kernel, RBF):
        return RBF(length_scale=length_scale)
    elif isinstance(kernel, Matern):
        return Matern(length_scale=length_scale, nu=kernel.nu)
    elif isinstance(kernel, RationalQuadratic):
        return RationalQuadratic(length_scale=length_scale, alpha=kernel.alpha)
    return None


def _optimize_n_restarts_optimizer(X_train, y_train, X_test, y_true, opts):
    """Optimize number of restarts for optimizer"""
    max_acc = -np.inf
    best_f1 = 0.0
    variable_array = np.array([0, 1, 2, 5, 10])
    best_val = variable_array[0]

    with tqdm(
        variable_array, desc="Optimizing Number of Restarts", leave=False
    ) as pbar:
        for v in pbar:
            test_opts = opts.copy()
            test_opts["n_restarts_optimizer"] = v

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_val = v

            pbar.set_postfix(
                {
                    "restarts": v,
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts["n_restarts_optimizer"] = best_val
    return opts, max_acc, best_f1


def _optimize_max_iter_predict(X_train, y_train, X_test, y_true, opts):
    """Optimize maximum iterations for prediction"""
    max_acc = -np.inf
    best_f1 = 0.0
    variable_array = np.array([100, 200, 500, 1000])
    best_val = variable_array[0]

    with tqdm(
        variable_array, desc="Optimizing Max Iterations Predict", leave=False
    ) as pbar:
        for v in pbar:
            test_opts = opts.copy()
            test_opts["max_iter_predict"] = v

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, test_opts
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

    opts["max_iter_predict"] = best_val
    return opts, max_acc, best_f1


def _optimize_warm_start(X_train, y_train, X_test, y_true, opts):
    """Optimize warm start parameter"""
    max_acc = -np.inf
    best_f1 = 0.0
    variable_array = [True, False]
    best_val = variable_array[0]

    with tqdm(variable_array, desc="Optimizing Warm Start", leave=False) as pbar:
        for v in pbar:
            test_opts = opts.copy()
            test_opts["warm_start"] = v

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_val = v

            pbar.set_postfix(
                {
                    "warm_start": str(v),
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts["warm_start"] = best_val
    return opts, max_acc, best_f1


def _optimize_copy_X_train(X_train, y_train, X_test, y_true, opts):
    """Optimize copy_X_train parameter"""
    max_acc = -np.inf
    best_f1 = 0.0
    variable_array = [True, False]
    best_val = variable_array[0]

    with tqdm(variable_array, desc="Optimizing Copy X Train", leave=False) as pbar:
        for v in pbar:
            test_opts = opts.copy()
            test_opts["copy_X_train"] = v

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_val = v

            pbar.set_postfix(
                {
                    "copy_X": str(v),
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts["copy_X_train"] = best_val
    return opts, max_acc, best_f1


def _optimize_multi_class(X_train, y_train, X_test, y_true, opts):
    """Optimize multi-class strategy"""
    max_acc = -np.inf
    best_f1 = 0.0
    variable_array = ["one_vs_rest", "one_vs_one"]
    best_val = variable_array[0]

    with tqdm(
        variable_array, desc="Optimizing Multi-Class Strategy", leave=False
    ) as pbar:
        for v in pbar:
            test_opts = opts.copy()
            test_opts["multi_class"] = v

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_val = v

            pbar.set_postfix(
                {
                    "strategy": v[:8],
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts["multi_class"] = best_val
    return opts, max_acc, best_f1


def _optimize_kernel_bounds(X_train, y_train, X_test, y_true, opts):
    """Optimize kernel hyperparameter bounds"""
    max_acc = -np.inf
    best_f1 = 0.0

    # Different bound ranges to test
    bound_options = [
        (1e-5, 1e5),  # Default wide range
        (1e-3, 1e3),  # Moderate range
        (1e-2, 1e2),  # Narrow range
        (1e-4, 1e4),  # Medium-wide range
    ]

    current_kernel = opts["kernel"]

    with tqdm(bound_options, desc="Optimizing Kernel Bounds", leave=False) as pbar:
        for bounds in pbar:
            new_kernel = _create_kernel_with_bounds(current_kernel, bounds)

            if new_kernel is None:
                pbar.set_postfix(
                    {"bounds": f"{bounds[0]:.0e}-{bounds[1]:.0e}", "status": "skipped"}
                )
                continue

            test_opts = opts.copy()
            test_opts["kernel"] = new_kernel

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                opts["kernel"] = new_kernel

            pbar.set_postfix(
                {
                    "bounds": f"{bounds[0]:.0e}-{bounds[1]:.0e}",
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best": f"{max_acc:.4f}",
                }
            )

    return opts, max_acc, best_f1


def _create_kernel_with_bounds(kernel, bounds):
    """Create kernel with specified bounds"""
    try:
        # Handle composite kernels
        if hasattr(kernel, "k1") and hasattr(kernel, "k2"):
            if isinstance(kernel.k1, (RBF, Matern, RationalQuadratic)):
                base_kernel = kernel.k1
                noise_kernel = kernel.k2
            elif isinstance(kernel.k2, (RBF, Matern, RationalQuadratic)):
                base_kernel = kernel.k2
                noise_kernel = kernel.k1
            else:
                return None

            new_base = _create_simple_kernel_with_bounds(base_kernel, bounds)
            return new_base + noise_kernel if new_base else None

        # Handle simple kernels
        return _create_simple_kernel_with_bounds(kernel, bounds)

    except Exception:
        return None


def _create_simple_kernel_with_bounds(kernel, bounds):
    """Create simple kernel with specified bounds"""
    if isinstance(kernel, RBF):
        return RBF(length_scale=kernel.length_scale, length_scale_bounds=bounds)
    elif isinstance(kernel, Matern):
        return Matern(
            length_scale=kernel.length_scale, length_scale_bounds=bounds, nu=kernel.nu
        )
    elif isinstance(kernel, RationalQuadratic):
        return RationalQuadratic(
            length_scale=kernel.length_scale,
            length_scale_bounds=bounds,
            alpha=kernel.alpha,
        )
    return None


def _optimize_gaussian_process(X_train, y_train, X_test, y_true, cycles=2):
    """
    Optimizes the hyperparameters for GaussianProcessClassifier.
    :param X_train: PCA transformed training data
    :param y_train: Flattened training labels
    :param X_test: PCA transformed test data
    :param y_true: True labels for the test data
    :param cycles: Number of optimization cycles
    """

    opts = {
        "kernel": 1.0 * RBF(1.0),
        "optimizer": "fmin_l_bfgs_b",
        "n_restarts_optimizer": 0,
        "max_iter_predict": 100,
        "warm_start": False,
        "copy_X_train": True,
        "random_state": 42,
        "multi_class": "one_vs_rest",
        "n_jobs": -1,
    }

    # Optimize hyperparameters
    ma_vec = []
    f1_vec = []

    # Main optimization loop with overall progress bar
    with tqdm(range(cycles), desc="Optimization Cycles", position=0) as cycle_pbar:
        for c in cycle_pbar:
            cycle_pbar.set_description(f"Optimization Cycle {c + 1}/{cycles}")

            opts, _, _ = _optimize_kernel_type(X_train, y_train, X_test, y_true, opts)
            opts, _, _ = _optimize_kernel_length_scale(
                X_train, y_train, X_test, y_true, opts
            )
            opts, _, _ = _optimize_n_restarts_optimizer(
                X_train, y_train, X_test, y_true, opts
            )
            opts, _, _ = _optimize_max_iter_predict(
                X_train, y_train, X_test, y_true, opts
            )
            opts, _, _ = _optimize_warm_start(X_train, y_train, X_test, y_true, opts)
            opts, _, _ = _optimize_copy_X_train(X_train, y_train, X_test, y_true, opts)
            opts, _, _ = _optimize_multi_class(X_train, y_train, X_test, y_true, opts)
            opts, ma, f1 = _optimize_kernel_bounds(
                X_train, y_train, X_test, y_true, opts
            )

            ma_vec.append(ma)
            f1_vec.append(f1)

            cycle_pbar.set_postfix(
                {
                    "accuracy": f"{ma:.4f}",
                    "f1": f"{f1:.4f}",
                    "best_overall": f"{max(ma_vec):.4f}",
                }
            )

    return opts, ma, f1, ma_vec, f1_vec
