from typing import Any

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, f1_score


# ----------------- RANDOM STATE -----------------
def _optimize_rs(
    X_train_pca,
    y_train_flat,
    X_test_pca,
    y_true,
    random_state=None,
    estimator=None,
    n_estimators=50,
    learning_rate=1.0,
) -> tuple[Any, float]:
    """
    Optimizes the random state for a Random Forest classifier.
    :param X_train_pca: PCA transformed training data
    :param y_train_flat: Flattened training labels
    :param X_test_pca: PCA transformed test data
    :param y_true: True labels for the test data
    """
    # Initialize variables
    print("Optimizing random state...")
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = np.arange(150)
    best_val = variable_array[0]
    for v in variable_array:
        # Define classifiers to test
        classifier = AdaBoostClassifier(
            random_state=v,
            estimator=None,
            n_estimators=50,
            learning_rate=1.0,
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
    random_state = best_val

    return random_state, max_acc


def _optimize_nest(
    X_train_pca,
    y_train_flat,
    X_test_pca,
    y_true,
    random_state=42,
    estimator=None,
    n_estimators=50,
    learning_rate=1.0,
) -> tuple[Any, float]:
    print("Optimizing # estimators...")

    # Initialize variables
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = np.arange(1, 150, 1)
    best_val = variable_array[0]

    # Iterate through variables
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = AdaBoostClassifier(
            random_state=random_state,
            n_estimators=v,
            estimator=None,
            learning_rate=1.0,
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
    n_estimators = best_val

    return n_estimators, max_acc


def _optimize_lr(
    X_train_pca,
    y_train_flat,
    X_test_pca,
    y_true,
    random_state=42,
    estimator=None,
    n_estimators=50,
    learning_rate=1.0,
) -> tuple[Any, float]:
    print("Optimizing learning rate...")

    # Initialize variables
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = np.arange(0.01, 2.0, 0.01)
    best_val = variable_array[0]

    # Iterate through variables
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = AdaBoostClassifier(
            random_state=random_state,
            n_estimators=n_estimators,
            estimator=None,
            learning_rate=v,
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
    learning_rate = best_val

    return learning_rate, max_acc


def _optimize_est(
    X_train_pca,
    y_train_flat,
    X_test_pca,
    y_true,
    random_state=42,
    estimator=None,
    n_estimators=50,
    learning_rate=1.0,
) -> tuple[Any, float]:
    print("Optimizing estimator...")

    # Initialize variables
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = np.arange(1, 150, 1)
    best_val = variable_array[0]

    # Iterate through variables
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = AdaBoostClassifier(
            random_state=random_state,
            n_estimators=n_estimators,
            estimator=v,
            learning_rate=1.0,
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
    estimator = best_val

    return estimator, max_acc


def optimize_adaboost(
    X_train_pca,
    y_train_flat,
    X_test_pca,
    y_true,
    random_state=None,
    estimator=None,
    n_estimators=50,
    learning_rate=1.0,
) -> float:

    # Shorten parameters
    Xtr_pca = X_train_pca
    ytr_flat = y_train_flat
    Xte_pca = X_test_pca

    # Optimize parameters
    rs, ma = _optimize_rs(Xtr_pca, ytr_flat, Xte_pca, y_true)
    print("Accuracy: ", ma)
    nest, ma = _optimize_nest(Xtr_pca, ytr_flat, Xte_pca, y_true, random_state=rs)
    print("Accuracy: ", ma)
    lr, ma = _optimize_lr(
        Xtr_pca, ytr_flat, Xte_pca, y_true, random_state=rs, n_estimators=nest
    )
    print("Accuracy: ", ma)
    est, ma = _optimize_est(
        Xtr_pca,
        ytr_flat,
        Xte_pca,
        y_true,
        random_state=rs,
        n_estimators=nest,
        learning_rate=lr,
    )
    print("Accuracy: ", ma)
    rs, ma = _optimize_rs(
        Xtr_pca,
        ytr_flat,
        Xte_pca,
        y_true,
        random_state=rs,
        n_estimators=nest,
        estimator=est,
        learning_rate=lr,
    )
    print("Accuracy: ", ma)
    nest, ma = _optimize_nest(
        Xtr_pca,
        ytr_flat,
        Xte_pca,
        y_true,
        random_state=rs,
        n_estimators=nest,
        estimator=est,
        learning_rate=lr,
    )
    print("Accuracy: ", ma)

    return ma
