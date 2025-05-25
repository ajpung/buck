from typing import Any

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score


# ----------------- RANDOM STATE -----------------
def _optimize_rs(
    X_train_pca,
    y_train_flat,
    X_test_pca,
    y_true,
    opts,
):
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = np.arange(800)
    best_val = variable_array[0]
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = RidgeClassifier(
            random_state=v,
            alpha=opts["alpha"],
            fit_intercept=opts["fit_intercept"],
            copy_X=opts["copy_X"],
            max_iter=opts["max_iter"],
            tol=opts["tol"],
            class_weight=opts["class_weight"],
            solver=opts["solver"],
            positive=opts["positive"],
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
            f1s = f1
            best_val = v

    # Store best value
    opts["random_state"] = best_val

    return opts, max_acc, f1s


def _optimize_alpha(
    X_train_pca,
    y_train_flat,
    X_test_pca,
    y_true,
    opts,
):
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = np.arange(0.01, 1.0, 0.01)
    best_val = variable_array[0]
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = RidgeClassifier(
            random_state=opts["random_state"],
            alpha=v,
            fit_intercept=opts["fit_intercept"],
            copy_X=opts["copy_X"],
            max_iter=opts["max_iter"],
            tol=opts["tol"],
            class_weight=opts["class_weight"],
            solver=opts["solver"],
            positive=opts["positive"],
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
            f1s = f1
            best_val = v

    # Store best value
    opts["alpha"] = best_val
    return opts, max_acc, f1s


def _optimize_fit_intercept(
    X_train_pca,
    y_train_flat,
    X_test_pca,
    y_true,
    opts,
):
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = [True, False]
    best_val = variable_array[0]
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = RidgeClassifier(
            random_state=opts["random_state"],
            alpha=opts["alpha"],
            fit_intercept=v,
            copy_X=opts["copy_X"],
            max_iter=opts["max_iter"],
            tol=opts["tol"],
            class_weight=opts["class_weight"],
            solver=opts["solver"],
            positive=opts["positive"],
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
            f1s = f1
            best_val = v

    # Store best value
    opts["fit_intercept"] = best_val

    return opts, max_acc, f1s


def _optimize_max_iter(
    X_train_pca,
    y_train_flat,
    X_test_pca,
    y_true,
    opts,
):
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = np.arange(100, 1000, 100)
    best_val = variable_array[0]
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = RidgeClassifier(
            random_state=opts["random_state"],
            alpha=opts["alpha"],
            fit_intercept=opts["fit_intercept"],
            copy_X=opts["copy_X"],
            max_iter=v,
            tol=opts["tol"],
            class_weight=opts["class_weight"],
            solver=opts["solver"],
            positive=opts["positive"],
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
            f1s = f1
            best_val = v

    # Store best value
    opts["max_iter"] = best_val

    return opts, max_acc, f1s


def _optimize_tol(
    X_train_pca,
    y_train_flat,
    X_test_pca,
    y_true,
    opts,
):
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = np.arange(0.0001, 0.01, 0.0001)
    best_val = variable_array[0]
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = RidgeClassifier(
            random_state=opts["random_state"],
            alpha=opts["alpha"],
            fit_intercept=opts["fit_intercept"],
            copy_X=opts["copy_X"],
            max_iter=opts["max_iter"],
            tol=v,
            class_weight=opts["class_weight"],
            solver=opts["solver"],
            positive=opts["positive"],
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
            f1s = f1
            best_val = v

    # Store best value
    opts["tol"] = best_val

    return opts, max_acc, f1s


def _optimize_class_weight(
    X_train_pca,
    y_train_flat,
    X_test_pca,
    y_true,
    opts,
):
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = ["balanced", None]
    best_val = variable_array[0]
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = RidgeClassifier(
            random_state=opts["random_state"],
            alpha=opts["alpha"],
            fit_intercept=opts["fit_intercept"],
            copy_X=opts["copy_X"],
            max_iter=opts["max_iter"],
            tol=opts["tol"],
            class_weight=v,
            solver=opts["solver"],
            positive=opts["positive"],
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
            f1s = f1
            best_val = v

    # Store best value
    opts["class_weight"] = best_val

    return opts, max_acc, f1s


def _optimize_solver(
    X_train_pca,
    y_train_flat,
    X_test_pca,
    y_true,
    opts,
):
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = [
        "auto",
        "sag",
        "saga",
        "lsqr",
        "svd",
        "cholesky",
        "sparse_cg",
        "lbfgs",
    ]
    best_val = variable_array[0]
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = RidgeClassifier(
            random_state=opts["random_state"],
            alpha=opts["alpha"],
            fit_intercept=opts["fit_intercept"],
            copy_X=opts["copy_X"],
            max_iter=opts["max_iter"],
            tol=opts["tol"],
            class_weight=opts["class_weight"],
            solver=v,
            positive=opts["positive"],
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
            f1s = f1
            best_val = v

    # Store best value
    opts["solver"] = best_val

    return opts, max_acc, f1s


def _optimize_ridge(X_train_pca, y_train_flat, X_test_pca, y_true, cycles=2):

    # Shorten parameters
    Xtr_pca = X_train_pca
    ytr_flat = y_train_flat
    Xte_pca = X_test_pca

    # Define optimals
    opts = {
        "random_state": None,
        "alpha": 1.0,
        "fit_intercept": True,
        "copy_X": True,
        "max_iter": None,
        "tol": 0.0001,
        "class_weight": None,
        "solver": "auto",
        "positive": False,
    }

    # Cyclically optimize hyperparameters
    ma_vec = []
    f1_vec = []
    for c in np.arange(cycles):
        opts, _, _ = _optimize_rs(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _, _ = _optimize_alpha(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _, _ = _optimize_fit_intercept(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _, _ = _optimize_max_iter(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _, _ = _optimize_tol(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _, _ = _optimize_class_weight(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, ma, f1 = _optimize_solver(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        ma_vec.append(ma)
        f1_vec.append(f1)

    return opts, ma, f1, ma_vec, f1_vec
