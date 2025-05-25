from typing import Any

import numpy as np
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.metrics import accuracy_score, f1_score


def _optimize_threshold(X_train, y_train, X_test, y_true, opts):
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = np.arange(0.1, 1.0, 0.1)
    best_val = variable_array[0]
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = SelfTrainingClassifier(
            estimator=opts["estimator"],
            threshold=v,
            criterion=opts["criterion"],
            k_best=opts["k_best"],
            max_iter=opts["max_iter"],
            verbose=opts["verbose"],
        )
        # Train the classifier
        classifier.fit(X_train, y_train)
        # Make predictions
        y_pred = classifier.predict(X_test)
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
    opts["threshold"] = best_val

    return opts, max_acc, f1s


def _optimize_crit(X_train, y_train, X_test, y_true, opts):
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = ["k_best", "threshold"]
    best_val = variable_array[0]
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = SelfTrainingClassifier(
            estimator=opts["estimator"],
            threshold=opts["threshold"],
            criterion=v,
            k_best=opts["k_best"],
            max_iter=opts["max_iter"],
            verbose=opts["verbose"],
        )
        # Train the classifier
        classifier.fit(X_train, y_train)
        # Make predictions
        y_pred = classifier.predict(X_test)
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
    opts["criterion"] = best_val

    return opts, max_acc, f1s


def _optimize_kbest(X_train, y_train, X_test, y_true, opts):
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = np.arange(1, 21, 1)
    best_val = variable_array[0]
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = SelfTrainingClassifier(
            estimator=opts["estimator"],
            threshold=opts["threshold"],
            criterion=opts["criterion"],
            k_best=v,
            max_iter=opts["max_iter"],
            verbose=opts["verbose"],
        )
        # Train the classifier
        classifier.fit(X_train, y_train)
        # Make predictions
        y_pred = classifier.predict(X_test)
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
    opts["k_best"] = best_val

    return opts, max_acc, f1s


def _optimize_maxiter(X_train, y_train, X_test, y_true, opts):
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = np.arange(1, 500, 5)
    best_val = variable_array[0]
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = SelfTrainingClassifier(
            estimator=opts["estimator"],
            threshold=opts["threshold"],
            criterion=opts["criterion"],
            k_best=opts["k_best"],
            max_iter=v,
            verbose=opts["verbose"],
        )
        # Train the classifier
        classifier.fit(X_train, y_train)
        # Make predictions
        y_pred = classifier.predict(X_test)
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


def _optimize_self_training(X_train, y_train, X_test, y_true, cycles=2):

    # Shorten parameters
    Xtr_pca = X_train
    ytr_flat = y_train
    Xte_pca = X_test

    # Define optimals
    opts = {
        "estimator": None,
        "threshold": 0.75,
        "criterion": "threshold",
        "k_best": 10,
        "max_iter": 10,
        "verbose": False,
    }

    # Cyclically optimize hyperparameters
    ma_vec = []
    f1_vec = []
    for c in np.arange(cycles):
        opts, _, _ = _optimize_threshold(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _, _ = _optimize_crit(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _, _ = _optimize_kbest(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, ma, f1 = _optimize_maxiter(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        ma_vec.append(ma)
        f1_vec.append(f1)

    return opts, ma, f1, ma_vec, f1_vec
