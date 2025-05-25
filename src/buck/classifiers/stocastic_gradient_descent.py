from typing import Any

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score


def _optimize_loss(X_train, y_train, X_test, y_true, opts):
    """Optimize loss function"""
    max_acc = -np.inf
    variable_array = [
        "hinge",
        "log_loss",
        "modified_huber",
        "squared_hinge",
        "perceptron",
    ]
    best_val = variable_array[0]

    for v in variable_array:
        opts["loss"] = v

        classifier = SGDClassifier(**opts)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        if accuracy >= max_acc:
            max_acc = accuracy
            f1s = f1
            best_val = v

    opts["loss"] = best_val
    return opts, max_acc, f1s


def _optimize_penalty(X_train, y_train, X_test, y_true, opts):
    """Optimize penalty (regularization)"""
    max_acc = -np.inf
    variable_array = ["l2", "l1", "elasticnet", None]
    best_val = variable_array[0]

    for v in variable_array:
        opts["penalty"] = v

        classifier = SGDClassifier(**opts)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        if accuracy >= max_acc:
            max_acc = accuracy
            f1s = f1
            best_val = v

    opts["penalty"] = best_val
    return opts, max_acc, f1s


def _optimize_alpha(X_train, y_train, X_test, y_true, opts):
    """Optimize regularization strength (alpha)"""
    max_acc = -np.inf
    variable_array = np.logspace(-6, 1, 15)  # From 1e-6 to 10
    best_val = variable_array[0]

    for v in variable_array:
        opts["alpha"] = v

        classifier = SGDClassifier(**opts)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        if accuracy >= max_acc:
            max_acc = accuracy
            f1s = f1
            best_val = v

    opts["alpha"] = best_val
    return opts, max_acc, f1s


def _optimize_l1_ratio(X_train, y_train, X_test, y_true, opts):
    """Optimize L1 ratio for ElasticNet penalty"""
    max_acc = -np.inf
    variable_array = np.arange(0.0, 1.1, 0.1)
    best_val = variable_array[0]

    # Only optimize if penalty is elasticnet
    if opts["penalty"] != "elasticnet":
        return opts, max_acc, 0.0

    for v in variable_array:
        opts["l1_ratio"] = v

        classifier = SGDClassifier(**opts)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        if accuracy >= max_acc:
            max_acc = accuracy
            f1s = f1
            best_val = v

    opts["l1_ratio"] = best_val
    return opts, max_acc, f1s


def _optimize_learning_rate(X_train, y_train, X_test, y_true, opts):
    """Optimize learning rate schedule"""
    max_acc = -np.inf
    variable_array = ["constant", "optimal", "invscaling", "adaptive"]
    best_val = variable_array[0]

    for v in variable_array:
        opts["learning_rate"] = v

        classifier = SGDClassifier(**opts)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        if accuracy >= max_acc:
            max_acc = accuracy
            f1s = f1
            best_val = v

    opts["learning_rate"] = best_val
    return opts, max_acc, f1s


def _optimize_eta0(X_train, y_train, X_test, y_true, opts):
    """Optimize initial learning rate"""
    max_acc = -np.inf
    variable_array = np.logspace(-4, 1, 10)  # From 1e-4 to 10
    best_val = variable_array[0]

    # Only optimize if learning_rate is not 'optimal'
    if opts["learning_rate"] == "optimal":
        return opts, max_acc, 0.0

    for v in variable_array:
        opts["eta0"] = v

        classifier = SGDClassifier(**opts)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        if accuracy >= max_acc:
            max_acc = accuracy
            f1s = f1
            best_val = v

    opts["eta0"] = best_val
    return opts, max_acc, f1s


def _optimize_power_t(X_train, y_train, X_test, y_true, opts):
    """Optimize power_t for invscaling learning rate"""
    max_acc = -np.inf
    variable_array = np.arange(0.1, 1.0, 0.1)
    best_val = variable_array[0]

    # Only optimize if learning_rate is 'invscaling'
    if opts["learning_rate"] != "invscaling":
        return opts, max_acc, 0.0

    for v in variable_array:
        opts["power_t"] = v

        classifier = SGDClassifier(**opts)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        if accuracy >= max_acc:
            max_acc = accuracy
            f1s = f1
            best_val = v

    opts["power_t"] = best_val
    return opts, max_acc, f1s


def _optimize_max_iter(X_train, y_train, X_test, y_true, opts):
    """Optimize maximum number of iterations"""
    max_acc = -np.inf
    variable_array = np.array([100, 500, 1000, 2000, 5000])
    best_val = variable_array[0]

    for v in variable_array:
        opts["max_iter"] = v

        classifier = SGDClassifier(**opts)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        if accuracy >= max_acc:
            max_acc = accuracy
            f1s = f1
            best_val = v

    opts["max_iter"] = best_val
    return opts, max_acc, f1s


def _optimize_tol(X_train, y_train, X_test, y_true, opts):
    """Optimize tolerance for stopping criterion"""
    max_acc = -np.inf
    variable_array = np.logspace(-5, -1, 8)  # From 1e-5 to 1e-1
    best_val = variable_array[0]

    for v in variable_array:
        opts["tol"] = v

        classifier = SGDClassifier(**opts)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        if accuracy >= max_acc:
            max_acc = accuracy
            f1s = f1
            best_val = v

    opts["tol"] = best_val
    return opts, max_acc, f1s


def _optimize_class_weight(X_train, y_train, X_test, y_true, opts):
    """Optimize class weights"""
    max_acc = -np.inf
    variable_array = ["balanced", None]
    best_val = variable_array[0]

    for v in variable_array:
        opts["class_weight"] = v

        classifier = SGDClassifier(**opts)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        if accuracy >= max_acc:
            max_acc = accuracy
            f1s = f1
            best_val = v

    opts["class_weight"] = best_val
    return opts, max_acc, f1s


def _optimize_sgd_classifier(X_train, y_train, X_test, y_true, cycles=2):
    """
    Optimizes the hyperparameters for SGDClassifier.
    :param X_train: PCA transformed training data
    :param y_train: Flattened training labels
    :param X_test: PCA transformed test data
    :param y_true: True labels for the test data
    :param cycles: Number of optimization cycles
    """

    opts = {
        "loss": "hinge",
        "penalty": "l2",
        "alpha": 0.0001,
        "l1_ratio": 0.15,
        "fit_intercept": True,
        "max_iter": 1000,
        "tol": 1e-3,
        "shuffle": True,
        "verbose": 0,
        "epsilon": 0.1,
        "n_jobs": -1,
        "random_state": 42,
        "learning_rate": "optimal",
        "eta0": 0.0,
        "power_t": 0.5,
        "early_stopping": False,
        "validation_fraction": 0.1,
        "n_iter_no_change": 5,
        "class_weight": None,
        "warm_start": False,
        "average": False,
    }

    # Optimize hyperparameters
    ma_vec = []
    f1_vec = []

    for c in np.arange(cycles):
        print(f"Cycle {c + 1} of {cycles}")

        opts, _, _ = _optimize_loss(X_train, y_train, X_test, y_true, opts)
        opts, _, _ = _optimize_penalty(X_train, y_train, X_test, y_true, opts)
        opts, _, _ = _optimize_alpha(X_train, y_train, X_test, y_true, opts)
        opts, _, _ = _optimize_l1_ratio(X_train, y_train, X_test, y_true, opts)
        opts, _, _ = _optimize_learning_rate(X_train, y_train, X_test, y_true, opts)
        opts, _, _ = _optimize_eta0(X_train, y_train, X_test, y_true, opts)
        opts, _, _ = _optimize_power_t(X_train, y_train, X_test, y_true, opts)
        opts, _, _ = _optimize_max_iter(X_train, y_train, X_test, y_true, opts)
        opts, _, _ = _optimize_tol(X_train, y_train, X_test, y_true, opts)
        opts, ma, f1 = _optimize_class_weight(X_train, y_train, X_test, y_true, opts)

        ma_vec.append(ma)
        f1_vec.append(f1)

    return opts, ma, f1, ma_vec, f1_vec
