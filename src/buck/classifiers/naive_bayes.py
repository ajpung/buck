from typing import Any
import warnings
import numpy as np
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


def _safe_evaluate_model(
    X_train, y_train, X_test, y_true, nb_type="gaussian", **kwargs
):
    """Safely evaluate a Naive Bayes model configuration"""
    try:
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
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        return accuracy, f1, True
    except Exception:
        return 0.0, 0.0, False


def _optimize_nb_type(X_train, y_train, X_test, y_true, opts):
    """Optimize Naive Bayes algorithm type"""
    max_acc = -np.inf
    best_f1 = 0.0

    # Different NB variants with their applicable scenarios
    nb_types = {
        "gaussian": {},  # For continuous features
        "multinomial": {"alpha": 1.0},  # For discrete features (requires non-negative)
        "bernoulli": {"alpha": 1.0, "binarize": 0.0},  # For binary features
        "complement": {"alpha": 1.0},  # Good for imbalanced datasets
    }

    best_type = "gaussian"
    best_params = {}

    with tqdm(nb_types.items(), desc="Optimizing NB Type", leave=False) as pbar:
        for nb_type, default_params in pbar:
            # Skip types that require non-negative features if we have negative values
            if nb_type in ["multinomial", "complement"] and np.any(X_train < 0):
                pbar.set_postfix(
                    {"type": nb_type, "status": "skipped (negative features)"}
                )
                continue

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
    """Optimize var_smoothing for Gaussian NB"""
    max_acc = -np.inf
    best_f1 = 0.0

    # Only optimize if using Gaussian NB
    if opts.get("nb_type", "gaussian") != "gaussian":
        return opts, max_acc, best_f1

    # More comprehensive range for variance smoothing
    variable_array = np.logspace(-12, -3, 15)  # 1e-12 to 1e-3
    best_val = variable_array[0]

    with tqdm(variable_array, desc="Optimizing Var Smoothing", leave=False) as pbar:
        for v in pbar:
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
    """Optimize alpha (smoothing parameter) for non-Gaussian NB"""
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

    # Range of alpha values for Laplace smoothing
    variable_array = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    best_val = variable_array[0]

    with tqdm(variable_array, desc="Optimizing Alpha", leave=False) as pbar:
        for v in pbar:
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
    """Optimize binarize threshold for Bernoulli NB"""
    max_acc = -np.inf
    best_f1 = 0.0

    # Only optimize if using Bernoulli NB
    if opts.get("nb_type", "gaussian") != "bernoulli":
        return opts, max_acc, best_f1

    # Range of binarization thresholds
    variable_array = [None, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    best_val = variable_array[0]

    with tqdm(
        variable_array, desc="Optimizing Binarize Threshold", leave=False
    ) as pbar:
        for v in pbar:
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
                    "binarize": str(v) if v is not None else "None",
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts["binarize"] = best_val
    return opts, max_acc, best_f1


def _optimize_fit_prior(X_train, y_train, X_test, y_true, opts):
    """Optimize fit_prior for applicable NB types"""
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
    best_val = variable_array[0]

    with tqdm(variable_array, desc="Optimizing Fit Prior", leave=False) as pbar:
        for v in pbar:
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


def _optimize_priors(X_train, y_train, X_test, y_true, opts):
    """Optimize class priors (basic optimization - uniform vs None)"""
    max_acc = -np.inf
    best_f1 = 0.0

    # Get unique classes and create uniform priors
    unique_classes = np.unique(y_train)
    n_classes = len(unique_classes)
    uniform_priors = np.ones(n_classes) / n_classes

    variable_array = [None, uniform_priors]
    best_val = variable_array[0]

    with tqdm(variable_array, desc="Optimizing Priors", leave=False) as pbar:
        for i, v in enumerate(pbar):
            test_opts = {k: v for k, v in opts.items() if k != "nb_type"}
            test_opts["priors"] = v

            accuracy, f1, success = _safe_evaluate_model(
                X_train,
                y_train,
                X_test,
                y_true,
                nb_type=opts.get("nb_type", "gaussian"),
                **test_opts,
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_val = v

            pbar.set_postfix(
                {
                    "priors": "None" if v is None else "uniform",
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts["priors"] = best_val
    return opts, max_acc, best_f1


def _optimize_naive_bayes(X_train, y_train, X_test, y_true, cycles=2):
    """
    Optimizes the hyperparameters for Naive Bayes classifiers.
    :param X_train: Training data
    :param y_train: Training labels
    :param X_test: Test data
    :param y_true: True labels for the test data
    :param cycles: Number of optimization cycles
    """

    # Define initial parameters
    opts = {
        "nb_type": "gaussian",
        "priors": None,
        "var_smoothing": 1e-9,
    }

    # Optimize hyperparameters
    ma_vec = []
    f1_vec = []

    # Main optimization loop with overall progress bar
    with tqdm(range(cycles), desc="Optimization Cycles", position=0) as cycle_pbar:
        for c in cycle_pbar:
            cycle_pbar.set_description(f"Naive Bayes Cycle {c + 1}/{cycles}")

            # First determine the best NB type
            opts, _, _ = _optimize_nb_type(X_train, y_train, X_test, y_true, opts)

            # Then optimize parameters specific to that type
            if opts["nb_type"] == "gaussian":
                opts, _, _ = _optimize_var_smoothing(
                    X_train, y_train, X_test, y_true, opts
                )
            else:
                opts, _, _ = _optimize_alpha(X_train, y_train, X_test, y_true, opts)
                opts, _, _ = _optimize_fit_prior(X_train, y_train, X_test, y_true, opts)

                if opts["nb_type"] == "bernoulli":
                    opts, _, _ = _optimize_binarize(
                        X_train, y_train, X_test, y_true, opts
                    )

            # Optimize priors for all types
            opts, ma, f1 = _optimize_priors(X_train, y_train, X_test, y_true, opts)

            ma_vec.append(ma)
            f1_vec.append(f1)

            # Display comprehensive cycle information
            cycle_pbar.set_postfix(
                {
                    "accuracy": f"{ma:.4f}",
                    "f1": f"{f1:.4f}",
                    "best_overall": f"{max(ma_vec):.4f}",
                    "type": opts["nb_type"][:8],
                    "var_smooth": (
                        f'{opts.get("var_smoothing", 0):.0e}'
                        if opts["nb_type"] == "gaussian"
                        else "N/A"
                    ),
                    "alpha": (
                        f'{opts.get("alpha", 0):.2f}' if "alpha" in opts else "N/A"
                    ),
                }
            )

    return opts, ma, f1, ma_vec, f1_vec
