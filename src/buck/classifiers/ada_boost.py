from typing import Any

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, f1_score


# ----------------- RANDOM STATE -----------------
def _optimize_rs(X_train_pca, y_train_flat, X_test_pca, y_true, opts):

    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = np.arange(150)
    best_val = variable_array[0]
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = AdaBoostClassifier(
            random_state=v,
            n_estimators=opts["n_estimators"],
            estimator=opts["estimator"],
            learning_rate=opts["learning_rate"],
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
            f1s = f1

    # Store best value
    opts["random_state"] = best_val

    return opts, max_acc, f1s


def _optimize_nest(X_train_pca, y_train_flat, X_test_pca, y_true, opts):
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
            random_state=opts["random_state"],
            n_estimators=v,
            estimator=opts["estimator"],
            learning_rate=opts["learning_rate"],
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
            f1s = f1

    # Store best value
    opts["n_estimators"] = best_val

    return opts, max_acc, f1s


def _optimize_lr(X_train_pca, y_train_flat, X_test_pca, y_true, opts):
    # Initialize variables
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = np.arange(0.001, 20.0, 0.01)
    best_val = variable_array[0]

    # Iterate through variables
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = AdaBoostClassifier(
            random_state=opts["random_state"],
            n_estimators=opts["n_estimators"],
            estimator=opts["estimator"],
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
            f1s = f1
            best_val = v

    # Store best value
    opts["learning_rate"] = best_val

    return opts, max_acc, f1s


def _optimize_ada_boost(X_train_pca, y_train_flat, X_test_pca, y_true, cycles=2):

    # Shorten parameters
    Xtr_pca = X_train_pca
    ytr_flat = y_train_flat
    Xte_pca = X_test_pca

    # Define optimals
    opts = {
        "random_state": None,
        "estimator": None,
        "n_estimators": 50,
        "learning_rate": 1.0,
    }

    # Cyclically optimize hyperparameters
    ma_vec = []
    f1_vec = []
    for c in np.arange(cycles):
        opts, _, _ = _optimize_rs(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _, _ = _optimize_nest(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, ma, f1 = _optimize_lr(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        ma_vec.append(ma)
        f1_vec.append(f1)

    return opts, ma, f1, ma_vec, f1_vec
