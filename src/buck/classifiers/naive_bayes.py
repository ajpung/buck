from typing import Any
import warnings
import numpy as np
import time
from tqdm.auto import tqdm
from sklearn.naive_bayes import (
    GaussianNB,
    MultinomialNB,
    BernoulliNB,
    ComplementNB,
    CategoricalNB,
)
from sklearn.metrics import accuracy_score, f1_score
from sklearn.exceptions import ConvergenceWarning

# Suppress warnings for cleaner progress bars
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Global efficiency controls (lighter for NB since it's already fast)
_max_time_per_step = 900  # 1 minute max per optimization step
_max_time_per_model = 900  # 5 seconds max per model evaluation (NB is very fast)
_min_accuracy_threshold = 0.10  # Stop if accuracy is consistently terrible
_consecutive_failures = 0
_max_consecutive_failures = 3


def _safe_evaluate_model(
    X_train, y_train, X_test, y_true, nb_type="gaussian", **kwargs
):
    """Safely evaluate a Naive Bayes model configuration with timeout protection"""
    global _consecutive_failures

    try:
        start_time = time.time()

        if nb_type == "gaussian":
            classifier = GaussianNB(**kwargs)
        elif nb_type == "multinomial":
            classifier = MultinomialNB(**kwargs)
        elif nb_type == "bernoulli":
            classifier = BernoulliNB(**kwargs)
        elif nb_type == "complement":
            classifier = ComplementNB(**kwargs)
        elif nb_type == "categorical":
            classifier = CategoricalNB(**kwargs)
        else:
            return 0.0, 0.0, False

        classifier.fit(X_train, y_train)

        ## Check if training took too long (very rare for NB)
        # if time.time() - start_time > _max_time_per_model:
        #    print(f"⏰ NB timeout after {_max_time_per_model}s")
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


def _get_smart_nb_types(X_train, y_train):
    """Get smart NB type selection based on data characteristics"""

    has_negative = np.any(X_train < 0)
    has_continuous = not np.all(X_train == X_train.astype(int))
    n_features = X_train.shape[1]
    n_samples = X_train.shape[0]

    # Smart type selection
    nb_types = {}

    # Always include Gaussian (works with any data)
    nb_types["gaussian"] = {}

    # Only include multinomial/complement if no negative values
    if not has_negative:
        nb_types["multinomial"] = {"alpha": 1.0}

        # ComplementNB is especially good for imbalanced datasets
        if len(np.unique(y_train)) > 2:  # Multi-class
            nb_types["complement"] = {"alpha": 1.0}

    # Include Bernoulli for smaller feature spaces or binary-like data
    if n_features <= 100 or not has_continuous:
        nb_types["bernoulli"] = {"alpha": 1.0, "binarize": 0.0}

    return nb_types


def _optimize_nb_type(X_train, y_train, X_test, y_true, opts):
    """Optimize Naive Bayes algorithm type with smart selection"""
    global _consecutive_failures
    start_time = time.time()
    _consecutive_failures = 0

    max_acc = -np.inf
    best_f1 = 0.0

    # Get smart NB types based on data characteristics
    nb_types = _get_smart_nb_types(X_train, y_train)

    best_type = "gaussian"
    best_params = {}

    with tqdm(nb_types.items(), desc="Optimizing NB Type", leave=False) as pbar:
        for nb_type, default_params in pbar:
            if _consecutive_failures >= _max_consecutive_failures:
                pbar.set_description("NB Type (POOR ACCURACY)")
                break

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, nb_type=nb_type, **default_params
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_type = nb_type
                best_params = default_params.copy()

            pbar.set_postfix(
                {
                    "type": nb_type[:8],
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts["nb_type"] = best_type
    opts.update(best_params)
    return opts, max_acc, best_f1


def _optimize_var_smoothing(X_train, y_train, X_test, y_true, opts):
    """Optimize var_smoothing for Gaussian NB with smart range"""
    global _consecutive_failures
    start_time = time.time()
    _consecutive_failures = 0

    max_acc = -np.inf
    best_f1 = 0.0

    # Only optimize if using Gaussian NB
    if opts.get("nb_type", "gaussian") != "gaussian":
        return opts, max_acc, best_f1

    # Smart range based on dataset characteristics
    n_samples, n_features = X_train.shape

    if n_samples < 1000:
        # Smaller datasets need more smoothing
        variable_array = [1e-12, 1e-10, 1e-9, 1e-8, 1e-6, 1e-4]
    elif n_samples < 10000:
        # Medium datasets - balanced range
        variable_array = [1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-5]
    else:
        # Large datasets can handle less smoothing
        variable_array = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-4]

    best_val = 1e-9  # Default sklearn value

    with tqdm(variable_array, desc="Optimizing Var Smoothing", leave=False) as pbar:
        for v in pbar:
            # Early stopping conditions
            if _consecutive_failures >= _max_consecutive_failures:
                pbar.set_description("Var Smoothing (POOR ACCURACY)")
                break

            test_opts = {k: v for k, v in opts.items() if k != "nb_type"}
            test_opts["var_smoothing"] = v

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, nb_type="gaussian", **test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_val = v

            pbar.set_postfix(
                {
                    "var_smooth": f"{v:.0e}",
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts["var_smoothing"] = best_val
    return opts, max_acc, best_f1


def _optimize_alpha(X_train, y_train, X_test, y_true, opts):
    """Optimize alpha (smoothing parameter) with smart range"""
    global _consecutive_failures
    start_time = time.time()
    _consecutive_failures = 0

    max_acc = -np.inf
    best_f1 = 0.0

    # Only optimize if using NB types that support alpha
    if opts.get("nb_type", "gaussian") not in [
        "multinomial",
        "bernoulli",
        "complement",
        "categorical",
    ]:
        return opts, max_acc, best_f1

    # Smart alpha range - more focused
    variable_array = [0.1, 0.5, 1.0, 2.0, 5.0]  # Reduced from 7 to 5 values
    best_val = 1.0  # Standard default

    with tqdm(variable_array, desc="Optimizing Alpha", leave=False) as pbar:
        for v in pbar:
            # Early stopping conditions
            if _consecutive_failures >= _max_consecutive_failures:
                pbar.set_description("Alpha (POOR ACCURACY)")
                break

            test_opts = {k: v for k, v in opts.items() if k != "nb_type"}
            test_opts["alpha"] = v

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, nb_type=opts["nb_type"], **test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_val = v

            pbar.set_postfix(
                {
                    "alpha": f"{v:.2f}",
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts["alpha"] = best_val
    return opts, max_acc, best_f1


def _optimize_binarize(X_train, y_train, X_test, y_true, opts):
    """Optimize binarize threshold for Bernoulli NB with smart selection"""
    global _consecutive_failures
    start_time = time.time()
    _consecutive_failures = 0

    max_acc = -np.inf
    best_f1 = 0.0

    # Only optimize if using Bernoulli NB
    if opts.get("nb_type", "gaussian") != "bernoulli":
        return opts, max_acc, best_f1

    # Smart threshold selection based on data distribution
    data_mean = np.mean(X_train)
    data_median = np.median(X_train)

    # Include data-driven thresholds
    variable_array = [None, 0.0, data_median, data_mean, 0.5]
    # Remove duplicates and sort
    variable_array = sorted(
        list(set([x for x in variable_array if x is None or (x >= 0 and x <= 1)]))
    )

    best_val = 0.0

    with tqdm(
        variable_array, desc="Optimizing Binarize Threshold", leave=False
    ) as pbar:
        for v in pbar:
            # Early stopping conditions
            if _consecutive_failures >= _max_consecutive_failures:
                pbar.set_description("Binarize (POOR ACCURACY)")
                break

            test_opts = {k: v for k, v in opts.items() if k != "nb_type"}
            test_opts["binarize"] = v

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, nb_type="bernoulli", **test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_val = v

            pbar.set_postfix(
                {
                    "binarize": f"{v:.3f}" if v is not None else "None",
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts["binarize"] = best_val
    return opts, max_acc, best_f1


def _optimize_fit_prior(X_train, y_train, X_test, y_true, opts):
    """Optimize fit_prior for applicable NB types"""
    global _consecutive_failures
    start_time = time.time()
    _consecutive_failures = 0

    max_acc = -np.inf
    best_f1 = 0.0

    # Only optimize for NB types that support fit_prior
    if opts.get("nb_type", "gaussian") not in [
        "multinomial",
        "bernoulli",
        "complement",
    ]:
        return opts, max_acc, best_f1

    variable_array = [True, False]
    best_val = True

    with tqdm(variable_array, desc="Optimizing Fit Prior", leave=False) as pbar:
        for v in pbar:
            # Early stopping conditions
            if _consecutive_failures >= _max_consecutive_failures:
                pbar.set_description("Fit Prior (POOR ACCURACY)")
                break

            test_opts = {k: v for k, v in opts.items() if k != "nb_type"}
            test_opts["fit_prior"] = v

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, nb_type=opts["nb_type"], **test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_val = v

            pbar.set_postfix(
                {
                    "fit_prior": str(v),
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts["fit_prior"] = best_val
    return opts, max_acc, best_f1


def _optimize_naive_bayes(X_train, y_train, X_test, y_true, cycles=2):
    """
    FAST optimized hyperparameters for Naive Bayes classifiers.
    :param X_train: Training data
    :param y_train: Training labels
    :param X_test: Test data
    :param y_true: True labels for the test data
    :param cycles: Number of optimization cycles
    """

    n_samples, n_features = X_train.shape
    print(f"Dataset: {n_samples} samples, {n_features} features")

    # Quick baseline check
    print("Running baseline check...")
    baseline_nb = GaussianNB()
    baseline_nb.fit(X_train, y_train)
    baseline_acc = baseline_nb.score(X_test, y_true)
    print(f"Baseline Gaussian NB accuracy: {baseline_acc:.4f}")

    # Data characteristics analysis
    has_negative = np.any(X_train < 0)
    has_continuous = not np.all(X_train == X_train.astype(int))
    n_classes = len(np.unique(y_train))

    print(f"Data characteristics:")
    print(f"  Has negative values: {has_negative}")
    print(f"  Has continuous features: {has_continuous}")
    print(f"  Number of classes: {n_classes}")

    if baseline_acc < 0.1:
        print("⚠️  WARNING: Very low baseline accuracy!")
        print("Consider checking feature scaling, data quality, or class distribution.")

    # Define initial parameters with smart defaults
    opts = {
        "nb_type": "gaussian",
        "priors": None,
        "var_smoothing": 1e-9,
    }

    # Track results
    ma_vec = []
    f1_vec = []

    # Main optimization loop
    with tqdm(
        range(cycles), desc="FAST Naive Bayes Optimization", position=0
    ) as cycle_pbar:
        for c in cycle_pbar:
            cycle_start_time = time.time()
            cycle_pbar.set_description(f"NB Cycle {c + 1}/{cycles}")

            # Core optimizations
            opts, _, _ = _optimize_nb_type(X_train, y_train, X_test, y_true, opts)

            # Type-specific optimizations
            if opts["nb_type"] == "gaussian":
                opts, _, _ = _optimize_var_smoothing(
                    X_train, y_train, X_test, y_true, opts
                )
            else:
                opts, _, _ = _optimize_alpha(X_train, y_train, X_test, y_true, opts)
                opts, _, _ = _optimize_fit_prior(X_train, y_train, X_test, y_true, opts)

                if opts["nb_type"] == "bernoulli":
                    opts, ma, f1 = _optimize_binarize(
                        X_train, y_train, X_test, y_true, opts
                    )
                else:
                    # Final evaluation for non-Bernoulli types
                    test_opts = {k: v for k, v in opts.items() if k != "nb_type"}
                    ma, f1, _ = _safe_evaluate_model(
                        X_train,
                        y_train,
                        X_test,
                        y_true,
                        nb_type=opts["nb_type"],
                        **test_opts,
                    )

            # If Gaussian, get final evaluation
            if opts["nb_type"] == "gaussian":
                test_opts = {k: v for k, v in opts.items() if k != "nb_type"}
                ma, f1, _ = _safe_evaluate_model(
                    X_train, y_train, X_test, y_true, nb_type="gaussian", **test_opts
                )

            ma_vec.append(ma)
            f1_vec.append(f1)

            cycle_time = time.time() - cycle_start_time

            # Display comprehensive cycle information
            cycle_pbar.set_postfix(
                {
                    "accuracy": f"{ma:.4f}",
                    "f1": f"{f1:.4f}",
                    "best_overall": f"{max(ma_vec):.4f}",
                    "cycle_time": f"{cycle_time:.1f}s",
                    "type": opts["nb_type"][:8],
                    "var_smooth": (
                        f'{opts.get("var_smoothing", 0):.0e}'
                        if opts["nb_type"] == "gaussian"
                        else "N/A"
                    ),
                    "alpha": (
                        f'{opts.get("alpha", 0):.2f}' if "alpha" in opts else "N/A"
                    ),
                    "baseline_beat": f"{ma - baseline_acc:+.4f}",
                }
            )

    return opts, ma, f1, ma_vec, f1_vec
