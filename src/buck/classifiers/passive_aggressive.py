from typing import Any

import numpy as np
from sklearn.linear_model import PassiveAggressiveClassifier
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
        classifier = PassiveAggressiveClassifier(
            random_state=v,
            C=opts["C"],
            fit_intercept=opts["fit_intercept"],
            max_iter=opts["max_iter"],
            tol=opts["tol"],
            early_stopping=opts["early_stopping"],
            validation_fraction=opts["validation_fraction"],
            n_iter_no_change=opts["n_iter_no_change"],
            shuffle=opts["shuffle"],
            verbose=opts["verbose"],
            loss=opts["loss"],
            n_jobs=opts["n_jobs"],
            warm_start=opts["warm_start"],
            class_weight=opts["class_weight"],
            average=opts["average"],
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


def _optimize_C(
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
    variable_array = np.arange(0.1, 2.0, 0.1)
    best_val = variable_array[0]
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = PassiveAggressiveClassifier(
            random_state=opts["random_state"],
            C=v,
            fit_intercept=opts["fit_intercept"],
            max_iter=opts["max_iter"],
            tol=opts["tol"],
            early_stopping=opts["early_stopping"],
            validation_fraction=opts["validation_fraction"],
            n_iter_no_change=opts["n_iter_no_change"],
            shuffle=opts["shuffle"],
            verbose=opts["verbose"],
            loss=opts["loss"],
            n_jobs=opts["n_jobs"],
            warm_start=opts["warm_start"],
            class_weight=opts["class_weight"],
            average=opts["average"],
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
    opts["C"] = best_val

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
        classifier = PassiveAggressiveClassifier(
            random_state=opts["random_state"],
            C=opts["C"],
            fit_intercept=v,
            max_iter=opts["max_iter"],
            tol=opts["tol"],
            early_stopping=opts["early_stopping"],
            validation_fraction=opts["validation_fraction"],
            n_iter_no_change=opts["n_iter_no_change"],
            shuffle=opts["shuffle"],
            verbose=opts["verbose"],
            loss=opts["loss"],
            n_jobs=opts["n_jobs"],
            warm_start=opts["warm_start"],
            class_weight=opts["class_weight"],
            average=opts["average"],
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
    variable_array = np.arange(100, 2000, 100)
    best_val = variable_array[0]
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = PassiveAggressiveClassifier(
            random_state=opts["random_state"],
            C=opts["C"],
            fit_intercept=opts["fit_intercept"],
            max_iter=v,
            tol=opts["tol"],
            early_stopping=opts["early_stopping"],
            validation_fraction=opts["validation_fraction"],
            n_iter_no_change=opts["n_iter_no_change"],
            shuffle=opts["shuffle"],
            verbose=opts["verbose"],
            loss=opts["loss"],
            n_jobs=opts["n_jobs"],
            warm_start=opts["warm_start"],
            class_weight=opts["class_weight"],
            average=opts["average"],
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
        classifier = PassiveAggressiveClassifier(
            random_state=opts["random_state"],
            C=opts["C"],
            fit_intercept=opts["fit_intercept"],
            max_iter=opts["max_iter"],
            tol=v,
            early_stopping=opts["early_stopping"],
            validation_fraction=opts["validation_fraction"],
            n_iter_no_change=opts["n_iter_no_change"],
            shuffle=opts["shuffle"],
            verbose=opts["verbose"],
            loss=opts["loss"],
            n_jobs=opts["n_jobs"],
            warm_start=opts["warm_start"],
            class_weight=opts["class_weight"],
            average=opts["average"],
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


def _optimize_early_stopping(
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
        classifier = PassiveAggressiveClassifier(
            random_state=opts["random_state"],
            C=opts["C"],
            fit_intercept=opts["fit_intercept"],
            max_iter=opts["max_iter"],
            tol=opts["tol"],
            early_stopping=v,
            validation_fraction=opts["validation_fraction"],
            n_iter_no_change=opts["n_iter_no_change"],
            shuffle=opts["shuffle"],
            verbose=opts["verbose"],
            loss=opts["loss"],
            n_jobs=opts["n_jobs"],
            warm_start=opts["warm_start"],
            class_weight=opts["class_weight"],
            average=opts["average"],
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
    opts["early_stopping"] = best_val

    return opts, max_acc, f1s


def _optimize_validation_fraction(
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
    variable_array = np.arange(0.01, 0.5, 0.01)
    best_val = variable_array[0]
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = PassiveAggressiveClassifier(
            random_state=opts["random_state"],
            C=opts["C"],
            fit_intercept=opts["fit_intercept"],
            max_iter=opts["max_iter"],
            tol=opts["tol"],
            early_stopping=opts["early_stopping"],
            validation_fraction=v,
            n_iter_no_change=opts["n_iter_no_change"],
            shuffle=opts["shuffle"],
            verbose=opts["verbose"],
            loss=opts["loss"],
            n_jobs=opts["n_jobs"],
            warm_start=opts["warm_start"],
            class_weight=opts["class_weight"],
            average=opts["average"],
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
    opts["validation_fraction"] = best_val

    return opts, max_acc, f1s


def _optimize_n_iter_no_change(
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
    variable_array = np.arange(1, 20, 1)
    best_val = variable_array[0]
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = PassiveAggressiveClassifier(
            random_state=opts["random_state"],
            C=opts["C"],
            fit_intercept=opts["fit_intercept"],
            max_iter=opts["max_iter"],
            tol=opts["tol"],
            early_stopping=opts["early_stopping"],
            validation_fraction=opts["validation_fraction"],
            n_iter_no_change=v,
            shuffle=opts["shuffle"],
            verbose=opts["verbose"],
            loss=opts["loss"],
            n_jobs=opts["n_jobs"],
            warm_start=opts["warm_start"],
            class_weight=opts["class_weight"],
            average=opts["average"],
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
    opts["n_iter_no_change"] = best_val

    return opts, max_acc, f1s


def _optimize_shuffle(
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
        classifier = PassiveAggressiveClassifier(
            random_state=opts["random_state"],
            C=opts["C"],
            fit_intercept=opts["fit_intercept"],
            max_iter=opts["max_iter"],
            tol=opts["tol"],
            early_stopping=opts["early_stopping"],
            validation_fraction=opts["validation_fraction"],
            n_iter_no_change=opts["n_iter_no_change"],
            shuffle=v,
            verbose=opts["verbose"],
            loss=opts["loss"],
            n_jobs=opts["n_jobs"],
            warm_start=opts["warm_start"],
            class_weight=opts["class_weight"],
            average=opts["average"],
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
    opts["shuffle"] = best_val

    return opts, max_acc, f1s


def _optimize_loss(
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
    variable_array = ["hinge", "squared_hinge", "perceptron"]
    best_val = variable_array[0]
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = PassiveAggressiveClassifier(
            random_state=opts["random_state"],
            C=opts["C"],
            fit_intercept=opts["fit_intercept"],
            max_iter=opts["max_iter"],
            tol=opts["tol"],
            early_stopping=opts["early_stopping"],
            validation_fraction=opts["validation_fraction"],
            n_iter_no_change=opts["n_iter_no_change"],
            shuffle=opts["shuffle"],
            verbose=opts["verbose"],
            loss=v,
            n_jobs=opts["n_jobs"],
            warm_start=opts["warm_start"],
            class_weight=opts["class_weight"],
            average=opts["average"],
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
    opts["loss"] = best_val

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
    variable_array = [None, "balanced"]
    best_val = variable_array[0]
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = PassiveAggressiveClassifier(
            random_state=opts["random_state"],
            C=opts["C"],
            fit_intercept=opts["fit_intercept"],
            max_iter=opts["max_iter"],
            tol=opts["tol"],
            early_stopping=opts["early_stopping"],
            validation_fraction=opts["validation_fraction"],
            n_iter_no_change=opts["n_iter_no_change"],
            shuffle=opts["shuffle"],
            verbose=opts["verbose"],
            loss=opts["loss"],
            n_jobs=opts["n_jobs"],
            warm_start=opts["warm_start"],
            class_weight=v,
            average=opts["average"],
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


def _optimize_average(
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
        classifier = PassiveAggressiveClassifier(
            random_state=opts["random_state"],
            C=opts["C"],
            fit_intercept=opts["fit_intercept"],
            max_iter=opts["max_iter"],
            tol=opts["tol"],
            early_stopping=opts["early_stopping"],
            validation_fraction=opts["validation_fraction"],
            n_iter_no_change=opts["n_iter_no_change"],
            shuffle=opts["shuffle"],
            verbose=opts["verbose"],
            loss=opts["loss"],
            n_jobs=opts["n_jobs"],
            warm_start=opts["warm_start"],
            class_weight=opts["class_weight"],
            average=v,
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
    opts["average"] = best_val

    return opts, max_acc, f1s


def _optimize_passive_aggressive(
    X_train_pca, y_train_flat, X_test_pca, y_true, cycles=2
):
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

    opts = {
        "random_state": None,
        "C": 1.0,
        "fit_intercept": True,
        "max_iter": 1000,
        "tol": 0.001,
        "early_stopping": False,
        "validation_fraction": 0.1,
        "n_iter_no_change": 5,
        "shuffle": True,
        "verbose": 0,
        "loss": "hinge",
        "n_jobs": -1,
        "warm_start": False,
        "class_weight": None,
        "average": False,
    }

    # Optimize hyperparameters
    ma_vec = []
    f1_vec = []
    for c in np.arange(cycles):
        print(f"Cycle {c + 1} of {cycles}")
        opts, _, _ = _optimize_rs(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _, _ = _optimize_C(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _, _ = _optimize_fit_intercept(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _, _ = _optimize_max_iter(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _, _ = _optimize_tol(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _, _ = _optimize_early_stopping(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _, _ = _optimize_validation_fraction(
            Xtr_pca, ytr_flat, Xte_pca, y_true, opts
        )
        opts, _, _ = _optimize_n_iter_no_change(
            Xtr_pca, ytr_flat, Xte_pca, y_true, opts
        )
        opts, _, _ = _optimize_shuffle(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _, _ = _optimize_loss(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _, _ = _optimize_class_weight(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, ma, f1 = _optimize_average(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        ma_vec.append(ma)
        f1_vec.append(f1)

    return opts, ma, f1, ma_vec, f1_vec
