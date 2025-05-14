from typing import Any

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score


# ----------------- RANDOM STATE -----------------
def _optimize_nn(
    X_train_pca,
    y_train_flat,
    X_test_pca,
    y_true,
    n_neighbors=5,
    weights="uniform",
    algorithm="auto",
    leaf_size=30,
    p=2,
    metric="minkowski",
    metric_params=None,
    n_jobs=-1,
) -> tuple[Any, float]:
    """
    Optimizes the random state for a Random Forest classifier.
    :param X_train_pca: PCA transformed training data
    :param y_train_flat: Flattened training labels
    :param X_test_pca: PCA transformed test data
    :param y_true: True labels for the test data
    """
    # Initialize variables
    print("Optimizing neighbors...")
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = np.arange(150)
    best_val = variable_array[0]
    for v in variable_array:
        # Define classifiers to test
        classifier = KNeighborsClassifier(
            n_neighbors=v,
            weights="uniform",
            algorithm="auto",
            leaf_size=30,
            p=2,
            metric="minkowski",
            metric_params=None,
            n_jobs=None,
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
    n_neighbors = best_val

    return n_neighbors, max_acc


def _optimize_wt(
    X_train_pca,
    y_train_flat,
    X_test_pca,
    y_true,
    n_neighbors,
    weights="uniform",
    algorithm="auto",
    leaf_size=30,
    p=2,
    metric="minkowski",
    metric_params=None,
    n_jobs=-1,
) -> tuple[Any, float]:
    """
    Optimizes the weights for a Random Forest classifier.
    :param X_train_pca: PCA transformed training data
    :param y_train_flat: Flattened training labels
    :param X_test_pca: PCA transformed test data
    :param y_true: True labels for the test data
    """
    # Initialize variables
    print("Optimizing weights...")
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
            n_neighbors=n_neighbors,
            weights=v,
            algorithm="auto",
            leaf_size=30,
            p=2,
            metric="minkowski",
            metric_params=None,
            n_jobs=None,
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
    weights = best_val

    return weights, max_acc


def _optimize_algo(
    X_train_pca,
    y_train_flat,
    X_test_pca,
    y_true,
    n_neighbors,
    weights,
    algorithm="auto",
    leaf_size=30,
    p=2,
    metric="minkowski",
    metric_params=None,
    n_jobs=-1,
) -> tuple[Any, float]:
    """
    Optimizes the algorithm for a Random Forest classifier.
    :param X_train_pca: PCA transformed training data
    :param y_train_flat: Flattened training labels
    :param X_test_pca: PCA transformed test data
    :param y_true: True labels for the test data
    """
    # Initialize variables
    print("Optimizing algorithm...")
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
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=v,
            leaf_size=30,
            p=2,
            metric="minkowski",
            metric_params=None,
            n_jobs=None,
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
    algorithm = best_val

    return algorithm, max_acc


def _optimize_ls(
    X_train_pca,
    y_train_flat,
    X_test_pca,
    y_true,
    n_neighbors,
    weights,
    algorithm,
    leaf_size=30,
    p=2,
    metric="minkowski",
    metric_params=None,
    n_jobs=-1,
) -> tuple[Any, float]:
    """
    Optimizes the leaf size for a Random Forest classifier.
    :param X_train_pca: PCA transformed training data
    :param y_train_flat: Flattened training labels
    :param X_test_pca: PCA transformed test data
    :param y_true: True labels for the test data
    """
    # Initialize variables
    print("Optimizing leaf size...")
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
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=v,
            p=2,
            metric="minkowski",
            metric_params=None,
            n_jobs=None,
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
    leaf_size = best_val

    return leaf_size, max_acc


def _optimize_p(
    X_train_pca,
    y_train_flat,
    X_test_pca,
    y_true,
    n_neighbors,
    weights,
    algorithm,
    leaf_size,
    p=2,
    metric="minkowski",
    metric_params=None,
    n_jobs=-1,
) -> tuple[Any, float]:
    """
    Optimizes the p for a Random Forest classifier.
    :param X_train_pca: PCA transformed training data
    :param y_train_flat: Flattened training labels
    :param X_test_pca: PCA transformed test data
    :param y_true: True labels for the test data
    """
    # Initialize variables
    print("Optimizing p...")
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
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=v,
            metric="minkowski",
            metric_params=None,
            n_jobs=None,
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
    p = best_val

    return p, max_acc


def _optimize_metric(
    X_train_pca,
    y_train_flat,
    X_test_pca,
    y_true,
    n_neighbors,
    weights,
    algorithm,
    leaf_size,
    p,
    metric="minkowski",
    metric_params=None,
    n_jobs=-1,
) -> tuple[Any, float]:
    """
    Optimizes the metric for a Random Forest classifier.
    :param X_train_pca: PCA transformed training data
    :param y_train_flat: Flattened training labels
    :param X_test_pca: PCA transformed test data
    :param y_true: True labels for the test data
    """
    # Initialize variables
    print("Optimizing metric...")
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = [
        "minkowski",
        "euclidean",
        "manhattan",
        "chebyshev",
        "hamming",
        "cityblock",
        "braycurtis",
        "canberra",
        "correlation",
        "cosine",
        "mahalanobis",
        "seuclidean",
        "sqeuclidean",
        "wminkowski",
        "yule",
        "matching",
        "jaccard",
        "dice",
        "kulsinski",
        "rogerstanimoto",
        "sokalmichener",
        "sokalsneath",
    ]
    best_val = variable_array[0]
    for i in np.arange(len(variable_array)):
        # Set value
        v = variable_array[i]
        # Define classifiers to test
        classifier = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            metric=v,
            metric_params=None,
            n_jobs=None,
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
    metric = best_val

    return metric, max_acc


def optimize_knn(X_train_pca, y_train_flat, X_test_pca, y_true):
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

    # Optimize hyperparameters
    nn, max_acc = _optimize_nn(Xtr_pca, ytr_flat, Xte_pca, y_true)
    wt, max_acc = _optimize_wt(Xtr_pca, ytr_flat, Xte_pca, y_true, nn)
    algo, max_acc = _optimize_algo(Xtr_pca, ytr_flat, Xte_pca, y_true, nn, wt)
    ls, max_acc = _optimize_ls(Xtr_pca, ytr_flat, Xte_pca, y_true, nn, wt, algo)
    p, max_acc = _optimize_p(Xtr_pca, ytr_flat, Xte_pca, y_true, nn, wt, algo, ls)
    mt, max_acc = _optimize_metric(
        Xtr_pca, ytr_flat, Xte_pca, y_true, nn, wt, algo, ls, p
    )
    nn, max_acc = _optimize_nn(Xtr_pca, ytr_flat, Xte_pca, y_true, nn, wt, algo, ls, p)
    wt, max_acc = _optimize_wt(Xtr_pca, ytr_flat, Xte_pca, y_true, nn, wt, algo, ls, p)
    print(f"Best accuracy: {max_acc}")
    print(
        f"Best parameters: n_neighbors={nn}, weights={wt}, algorithm={algo}, leaf_size={ls}, p={p}, metric={mt}"
    )
