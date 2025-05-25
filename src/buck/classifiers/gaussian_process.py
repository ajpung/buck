from typing import Any

import numpy as np
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


def _optimize_kernel_type(X_train, y_train, X_test, y_true, opts):
    """Optimize kernel type"""
    max_acc = -np.inf

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

    for kernel_name, kernel in kernels.items():
        try:
            opts["kernel"] = kernel

            classifier = GaussianProcessClassifier(**opts)
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

            if accuracy >= max_acc:
                max_acc = accuracy
                f1s = f1
                best_kernel = kernel_name
                opts["kernel"] = kernel
        except:
            # Skip problematic kernels
            continue

    return opts, max_acc, f1s


def _optimize_kernel_length_scale(X_train, y_train, X_test, y_true, opts):
    """Optimize kernel length scale"""
    max_acc = -np.inf
    variable_array = np.logspace(-2, 2, 8)  # From 0.01 to 100
    best_val = variable_array[0]

    # Get current kernel type to modify its length scale
    current_kernel = opts["kernel"]

    for v in variable_array:
        try:
            # Create new kernel with updated length scale
            if hasattr(current_kernel, "k1") and hasattr(
                current_kernel.k1, "length_scale"
            ):
                # For composite kernels (with WhiteKernel)
                if isinstance(current_kernel.k1, RBF):
                    new_kernel = current_kernel.k1.get_params()["k2"].__class__(
                        noise_level=current_kernel.k1.get_params()["k2"].noise_level
                    ) + current_kernel.k1.get_params()["k1"].__class__(length_scale=v)
                elif isinstance(current_kernel.k1, Matern):
                    nu_val = current_kernel.k1.nu
                    new_kernel = current_kernel.k1.get_params()["k2"].__class__(
                        noise_level=current_kernel.k1.get_params()["k2"].noise_level
                    ) + current_kernel.k1.__class__(length_scale=v, nu=nu_val)
                else:
                    continue
            elif hasattr(current_kernel, "length_scale"):
                # For simple kernels
                if isinstance(current_kernel, RBF):
                    new_kernel = RBF(length_scale=v)
                elif isinstance(current_kernel, Matern):
                    nu_val = current_kernel.nu
                    new_kernel = Matern(length_scale=v, nu=nu_val)
                elif isinstance(current_kernel, RationalQuadratic):
                    alpha_val = current_kernel.alpha
                    new_kernel = RationalQuadratic(length_scale=v, alpha=alpha_val)
                else:
                    continue
            else:
                continue

            opts["kernel"] = new_kernel

            classifier = GaussianProcessClassifier(**opts)
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

            if accuracy >= max_acc:
                max_acc = accuracy
                f1s = f1
                best_val = v
        except:
            continue

    return opts, max_acc, f1s


def _optimize_n_restarts_optimizer(X_train, y_train, X_test, y_true, opts):
    """Optimize number of restarts for optimizer"""
    max_acc = -np.inf
    variable_array = np.array([0, 1, 2, 5, 10])
    best_val = variable_array[0]

    for v in variable_array:
        opts["n_restarts_optimizer"] = v

        classifier = GaussianProcessClassifier(**opts)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        if accuracy >= max_acc:
            max_acc = accuracy
            f1s = f1
            best_val = v

    opts["n_restarts_optimizer"] = best_val
    return opts, max_acc, f1s


def _optimize_max_iter_predict(X_train, y_train, X_test, y_true, opts):
    """Optimize maximum iterations for prediction"""
    max_acc = -np.inf
    variable_array = np.array([100, 200, 500, 1000])
    best_val = variable_array[0]

    for v in variable_array:
        opts["max_iter_predict"] = v

        classifier = GaussianProcessClassifier(**opts)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        if accuracy >= max_acc:
            max_acc = accuracy
            f1s = f1
            best_val = v

    opts["max_iter_predict"] = best_val
    return opts, max_acc, f1s


def _optimize_warm_start(X_train, y_train, X_test, y_true, opts):
    """Optimize warm start parameter"""
    max_acc = -np.inf
    variable_array = [True, False]
    best_val = variable_array[0]

    for v in variable_array:
        opts["warm_start"] = v

        classifier = GaussianProcessClassifier(**opts)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        if accuracy >= max_acc:
            max_acc = accuracy
            f1s = f1
            best_val = v

    opts["warm_start"] = best_val
    return opts, max_acc, f1s


def _optimize_copy_X_train(X_train, y_train, X_test, y_true, opts):
    """Optimize copy_X_train parameter"""
    max_acc = -np.inf
    variable_array = [True, False]
    best_val = variable_array[0]

    for v in variable_array:
        opts["copy_X_train"] = v

        classifier = GaussianProcessClassifier(**opts)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        if accuracy >= max_acc:
            max_acc = accuracy
            f1s = f1
            best_val = v

    opts["copy_X_train"] = best_val
    return opts, max_acc, f1s


def _optimize_multi_class(X_train, y_train, X_test, y_true, opts):
    """Optimize multi-class strategy"""
    max_acc = -np.inf
    variable_array = ["one_vs_rest", "one_vs_one"]
    best_val = variable_array[0]

    for v in variable_array:
        opts["multi_class"] = v

        classifier = GaussianProcessClassifier(**opts)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        if accuracy >= max_acc:
            max_acc = accuracy
            f1s = f1
            best_val = v

    opts["multi_class"] = best_val
    return opts, max_acc, f1s


def _optimize_kernel_bounds(X_train, y_train, X_test, y_true, opts):
    """Optimize kernel hyperparameter bounds"""
    max_acc = -np.inf

    # Different bound ranges to test
    bound_options = [
        (1e-5, 1e5),  # Default wide range
        (1e-3, 1e3),  # Moderate range
        (1e-2, 1e2),  # Narrow range
        (1e-4, 1e4),  # Medium-wide range
    ]

    best_bounds = bound_options[0]
    current_kernel = opts["kernel"]

    for bounds in bound_options:
        try:
            # Create new kernel with updated bounds
            if isinstance(current_kernel, RBF):
                new_kernel = RBF(length_scale=1.0, length_scale_bounds=bounds)
            elif isinstance(current_kernel, Matern):
                nu_val = getattr(current_kernel, "nu", 1.5)
                new_kernel = Matern(
                    length_scale=1.0, length_scale_bounds=bounds, nu=nu_val
                )
            elif isinstance(current_kernel, RationalQuadratic):
                new_kernel = RationalQuadratic(
                    length_scale=1.0, length_scale_bounds=bounds, alpha=1.0
                )
            else:
                continue

            opts["kernel"] = new_kernel

            classifier = GaussianProcessClassifier(**opts)
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

            if accuracy >= max_acc:
                max_acc = accuracy
                f1s = f1
                best_bounds = bounds
        except:
            continue

    return opts, max_acc, f1s


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

    for c in np.arange(cycles):
        print(f"Cycle {c + 1} of {cycles}")

        opts, _, _ = _optimize_kernel_type(X_train, y_train, X_test, y_true, opts)
        opts, _, _ = _optimize_kernel_length_scale(
            X_train, y_train, X_test, y_true, opts
        )
        opts, _, _ = _optimize_n_restarts_optimizer(
            X_train, y_train, X_test, y_true, opts
        )
        opts, _, _ = _optimize_max_iter_predict(X_train, y_train, X_test, y_true, opts)
        opts, _, _ = _optimize_warm_start(X_train, y_train, X_test, y_true, opts)
        opts, _, _ = _optimize_copy_X_train(X_train, y_train, X_test, y_true, opts)
        opts, _, _ = _optimize_multi_class(X_train, y_train, X_test, y_true, opts)
        opts, ma, f1 = _optimize_kernel_bounds(X_train, y_train, X_test, y_true, opts)

        ma_vec.append(ma)
        f1_vec.append(f1)

    return opts, ma, f1, ma_vec, f1_vec
