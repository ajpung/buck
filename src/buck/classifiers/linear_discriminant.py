from typing import Any

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score


# ----------------- RANDOM STATE -----------------
def _optimize_sl(
    X_train_pca,
    y_train_flat,
    X_test_pca,
    y_true,
    opts,
) -> tuple[Any, float]:
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = ["svd", "lsqr", "eigen"]
    best_val = variable_array[0]
    for v in variable_array:
        # Define classifiers to test
        classifier = LinearDiscriminantAnalysis(
            solver=v,
            shrinkage=opts["shrinkage"],
            priors=opts["priors"],
            n_components=opts["n_components"],
            store_covariance=opts["store_covariance"],
            tol=opts["tol"],
            covariance_estimator=opts["covariance_estimator"],
        )
        # Train the classifier
        classifier.fit(X_train_pca, y_train_flat)
        # Make predictions
        y_pred = classifier.predict(X_test_pca)
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        ac_vec.append(accuracy)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        f1_vec.append(f1)
        # Return index
        if accuracy >= max_acc:
            max_acc = accuracy
            best_val = v

    # Store best value
    opts["solver"] = best_val

    return opts, max_acc


def _optimize_sh(
    X_train_pca,
    y_train_flat,
    X_test_pca,
    y_true,
    opts,
) -> tuple[Any, float]:
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = np.arange(0, 1.1, 0.1)
    variable_array = np.append(variable_array.astype(object), ["auto", None])  # type: ignore
    best_val = variable_array[0]
    for v in variable_array:
        # Define classifiers to test
        classifier = LinearDiscriminantAnalysis(
            solver=opts["solver"],
            shrinkage=v,
            priors=opts["priors"],
            n_components=opts["n_components"],
            store_covariance=opts["store_covariance"],
            tol=opts["tol"],
            covariance_estimator=opts["covariance_estimator"],
        )
        # Train the classifier
        classifier.fit(X_train_pca, y_train_flat)
        # Make predictions
        y_pred = classifier.predict(X_test_pca)
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        ac_vec.append(accuracy)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        f1_vec.append(f1)
        # Return index
        if accuracy >= max_acc:
            max_acc = accuracy
            best_val = v

    # Store best value
    opts["shrinkage"] = best_val

    return opts, max_acc


def _optimize_nc(
    X_train_pca,
    y_train_flat,
    X_test_pca,
    y_true,
    opts,
) -> tuple[Any, float]:
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = np.arange(1, 20, 1)
    variable_array = np.append(variable_array.astype(object), None)  # type: ignore
    best_val = variable_array[0]
    for v in variable_array:
        # Define classifiers to test
        classifier = LinearDiscriminantAnalysis(
            solver=opts["solver"],
            shrinkage=v,
            priors=opts["priors"],
            n_components=opts["n_components"],
            store_covariance=opts["store_covariance"],
            tol=opts["tol"],
            covariance_estimator=opts["covariance_estimator"],
        )
        # Train the classifier
        classifier.fit(X_train_pca, y_train_flat)
        # Make predictions
        y_pred = classifier.predict(X_test_pca)
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        ac_vec.append(accuracy)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        f1_vec.append(f1)
        # Return index
        if accuracy >= max_acc:
            max_acc = accuracy
            best_val = v

    # Store best value
    opts["n_components"] = best_val

    return opts, max_acc


def _optimize_linear_discriminant(
    X_train_pca, y_train_flat, X_test_pca, y_true, cycles=2
) -> tuple[dict[str, float | bool | str | None], list[float]]:
    """
    Optimizes the hyperparameters for a Random Forest classifier.
    :param X_train_pca: PCA transformed training data
    :param y_train_flat: Flattened training labels
    :param X_test_pca: PCA transformed test data
    :param y_true: True labels for the test data
    """
    # Shorten parameters
    Xtr_pca = X_train_pca
    ytr_flat = y_train_flat
    Xte_pca = X_test_pca

    # Store optimals
    opts = {
        "solver": "svd",
        "shrinkage": None,
        "priors": None,
        "n_components": None,
        "store_covariance": False,
        "tol": 0.0001,
        "covariance_estimator": None,
    }

    # Optimize hyperparameters
    ma_vec = []
    for c in np.arange(cycles):
        opts, _ = _optimize_sl(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _ = _optimize_sh(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, ma = _optimize_nc(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        ma_vec.append(ma)

    return opts, ma_vec
