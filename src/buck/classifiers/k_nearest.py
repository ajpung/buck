from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier


# ----------------- RANDOM STATE -----------------
def _optimize_nn(
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
    variable_array = np.arange(1, 200)
    best_val = variable_array[0]
    for v in variable_array:
        # Define classifiers to test
        classifier = KNeighborsClassifier(
            n_neighbors=v,
            weights=opts["weights"],
            algorithm=opts["algorithm"],
            leaf_size=opts["leaf_size"],
            p=opts["p"],
            metric=opts["metric"],
            metric_params=opts["metric_params"],
            n_jobs=opts["n_jobs"],
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
    opts["n_neighbors"] = best_val

    return opts, max_acc


def _optimize_wt(
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
    variable_array = ["uniform", "distance", None]
    best_val = variable_array[0]
    for i in np.arange(len(variable_array)):
        # Set value
        v = variable_array[i]
        # Define classifiers to test
        classifier = KNeighborsClassifier(
            n_neighbors=opts["n_neighbors"],
            weights=v,
            algorithm=opts["algorithm"],
            leaf_size=opts["leaf_size"],
            p=opts["p"],
            metric=opts["metric"],
            metric_params=opts["metric_params"],
            n_jobs=opts["n_jobs"],
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
    opts["weights"] = best_val

    return opts, max_acc


def _optimize_algo(
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
    variable_array = ["auto", "ball_tree", "kd_tree", "brute"]
    best_val = variable_array[0]
    for i in np.arange(len(variable_array)):
        # Set value
        v = variable_array[i]
        # Define classifiers to test
        classifier = KNeighborsClassifier(
            n_neighbors=opts["n_neighbors"],
            weights=opts["weights"],
            algorithm=v,
            leaf_size=opts["leaf_size"],
            p=opts["p"],
            metric=opts["metric"],
            metric_params=opts["metric_params"],
            n_jobs=opts["n_jobs"],
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
    opts["algorithm"] = best_val

    return opts, max_acc


def _optimize_ls(
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
    variable_array = np.arange(1, 100)
    best_val = variable_array[0]
    for i in np.arange(len(variable_array)):
        # Set value
        v = variable_array[i]
        # Define classifiers to test
        classifier = KNeighborsClassifier(
            n_neighbors=opts["n_neighbors"],
            weights=opts["weights"],
            algorithm=opts["algorithm"],
            leaf_size=v,
            p=opts["p"],
            metric=opts["metric"],
            metric_params=opts["metric_params"],
            n_jobs=opts["n_jobs"],
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
    opts["leaf_size"] = best_val

    return opts, max_acc


def _optimize_p(
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
    variable_array = np.arange(1, 10, 0.2)
    best_val = variable_array[0]
    for i in np.arange(len(variable_array)):
        # Set value
        v = variable_array[i]
        # Define classifiers to test
        classifier = KNeighborsClassifier(
            n_neighbors=opts["n_neighbors"],
            weights=opts["weights"],
            algorithm=opts["algorithm"],
            leaf_size=opts["leaf_size"],
            p=v,
            metric=opts["metric"],
            metric_params=opts["metric_params"],
            n_jobs=opts["n_jobs"],
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
    opts["p"] = best_val

    return opts, max_acc


def _optimize_metric(
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

    "nan_euclidean",
    "p",
    "sqeuclidean",
    "russellrao",
    "l1",
    "l2"

    variable_array = [
        "braycurtis",
        "canberra",
        "chebyshev",
        "cityblock",
        "correlation",
        "cosine",
        "dice",
        "euclidean",
        "hamming",
        "jaccard",
        "manhattan",
        "minkowski",
        "rogerstanimoto",
        "russellrao",
        "sokalmichener",
        "sokalsneath",
        "yule",
    ]
    best_val = variable_array[0]
    for i in np.arange(len(variable_array)):
        # Set value
        v = variable_array[i]
        # Define classifiers to test
        classifier = KNeighborsClassifier(
            n_neighbors=opts["n_neighbors"],
            weights=opts["weights"],
            algorithm=opts["algorithm"],
            leaf_size=opts["leaf_size"],
            p=opts["p"],
            metric=v,
            metric_params=opts["metric_params"],
            n_jobs=opts["n_jobs"],
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
    opts["metric"] = best_val

    return opts, max_acc


def _optimize_knn(X_train_pca, y_train_flat, X_test_pca, y_true, cycles=2):
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
        "n_neighbors": 5,
        "weights": "uniform",
        "algorithm": "auto",
        "leaf_size": 30,
        "p": 2,
        "metric": "minkowski",
        "metric_params": None,
        "n_jobs": None,
    }

    # Optimize hyperparameters
    ma_vec = []
    for c in np.arange(cycles):
        opts, _ = _optimize_nn(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _ = _optimize_wt(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _ = _optimize_algo(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _ = _optimize_ls(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _ = _optimize_p(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, ma = _optimize_metric(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        ma_vec.append(ma)

    return opts, ma_vec
