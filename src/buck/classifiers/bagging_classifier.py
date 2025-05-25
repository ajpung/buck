from typing import Any

import numpy as np
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


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
        classifier = BaggingClassifier(
            random_state=v,
            estimator=opts["estimator"],
            n_estimators=opts["n_estimators"],
            max_samples=opts["max_samples"],
            max_features=opts["max_features"],
            bootstrap=opts["bootstrap"],
            bootstrap_features=opts["bootstrap_features"],
            oob_score=opts["oob_score"],
            warm_start=opts["warm_start"],
            n_jobs=opts["n_jobs"],
            verbose=opts["verbose"],
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


def _optimize_est(X_train_pca, y_train_flat, X_test_pca, y_true, opts):
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = [SVC(), RandomForestClassifier(), DecisionTreeClassifier()]
    best_val = variable_array[0]
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = BaggingClassifier(
            random_state=opts["random_state"],
            estimator=v,
            n_estimators=opts["n_estimators"],
            max_samples=opts["max_samples"],
            max_features=opts["max_features"],
            bootstrap=opts["bootstrap"],
            bootstrap_features=opts["bootstrap_features"],
            oob_score=opts["oob_score"],
            warm_start=opts["warm_start"],
            n_jobs=opts["n_jobs"],
            verbose=opts["verbose"],
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
    opts["estimator"] = best_val

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
        classifier = BaggingClassifier(
            random_state=opts["random_state"],
            estimator=opts["estimator"],
            n_estimators=v,
            max_samples=opts["max_samples"],
            max_features=opts["max_features"],
            bootstrap=opts["bootstrap"],
            bootstrap_features=opts["bootstrap_features"],
            oob_score=opts["oob_score"],
            warm_start=opts["warm_start"],
            n_jobs=opts["n_jobs"],
            verbose=opts["verbose"],
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
    opts["n_estimators"] = best_val

    return opts, max_acc, f1s


def _optimize_maxs(X_train_pca, y_train_flat, X_test_pca, y_true, opts):
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
        classifier = BaggingClassifier(
            random_state=opts["random_state"],
            estimator=opts["estimator"],
            n_estimators=opts["n_estimators"],
            max_samples=v,
            max_features=opts["max_features"],
            bootstrap=opts["bootstrap"],
            bootstrap_features=opts["bootstrap_features"],
            oob_score=opts["oob_score"],
            warm_start=opts["warm_start"],
            n_jobs=opts["n_jobs"],
            verbose=opts["verbose"],
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
    opts["max_samples"] = best_val

    return opts, max_acc, f1s


def _optimize_maxf(X_train_pca, y_train_flat, X_test_pca, y_true, opts):
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
        classifier = BaggingClassifier(
            random_state=opts["random_state"],
            estimator=opts["estimator"],
            n_estimators=opts["n_estimators"],
            max_samples=opts["max_samples"],
            max_features=v,
            bootstrap=opts["bootstrap"],
            bootstrap_features=opts["bootstrap_features"],
            oob_score=opts["oob_score"],
            warm_start=opts["warm_start"],
            n_jobs=opts["n_jobs"],
            verbose=opts["verbose"],
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
    opts["max_features"] = best_val

    return opts, max_acc, f1s


def _optimize_boot(X_train_pca, y_train_flat, X_test_pca, y_true, opts):
    # Initialize variables
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = [True, False]
    best_val = variable_array[0]

    # Iterate through variables
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = BaggingClassifier(
            random_state=opts["random_state"],
            estimator=opts["estimator"],
            n_estimators=opts["n_estimators"],
            max_samples=opts["max_samples"],
            max_features=opts["max_features"],
            bootstrap=v,
            bootstrap_features=opts["bootstrap_features"],
            oob_score=opts["oob_score"],
            warm_start=opts["warm_start"],
            n_jobs=opts["n_jobs"],
            verbose=opts["verbose"],
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
    opts["bootstrap"] = best_val

    return opts, max_acc, f1s


def _optimize_bootf(X_train_pca, y_train_flat, X_test_pca, y_true, opts):
    # Initialize variables
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = [True, False]
    best_val = variable_array[0]

    # Iterate through variables
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = BaggingClassifier(
            random_state=opts["random_state"],
            estimator=opts["estimator"],
            n_estimators=opts["n_estimators"],
            max_samples=opts["max_samples"],
            max_features=opts["max_features"],
            bootstrap=opts["bootstrap"],
            bootstrap_features=v,
            oob_score=opts["oob_score"],
            warm_start=opts["warm_start"],
            n_jobs=opts["n_jobs"],
            verbose=opts["verbose"],
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
    opts["bootstrap_features"] = best_val

    return opts, max_acc, f1s


def _optimize_oob(X_train_pca, y_train_flat, X_test_pca, y_true, opts):
    # Initialize variables
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = [True, False]
    best_val = variable_array[0]

    # Iterate through variables
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = BaggingClassifier(
            random_state=opts["random_state"],
            estimator=opts["estimator"],
            n_estimators=opts["n_estimators"],
            max_samples=opts["max_samples"],
            max_features=opts["max_features"],
            bootstrap=opts["bootstrap"],
            bootstrap_features=opts["bootstrap_features"],
            oob_score=v,
            warm_start=opts["warm_start"],
            n_jobs=opts["n_jobs"],
            verbose=opts["verbose"],
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
    opts["oob_score"] = best_val

    return opts, max_acc, f1s


def _optimize_warm(X_train_pca, y_train_flat, X_test_pca, y_true, opts):
    # Initialize variables
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = [True, False]
    best_val = variable_array[0]

    # Iterate through variables
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = BaggingClassifier(
            random_state=opts["random_state"],
            estimator=opts["estimator"],
            n_estimators=opts["n_estimators"],
            max_samples=opts["max_samples"],
            max_features=opts["max_features"],
            bootstrap=opts["bootstrap"],
            bootstrap_features=opts["bootstrap_features"],
            oob_score=opts["oob_score"],
            warm_start=v,
            n_jobs=opts["n_jobs"],
            verbose=opts["verbose"],
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
    opts["warm_start"] = best_val

    return opts, max_acc, f1s


def _optimize_bagging(X_train, y_train, X_test, y_true, cycles=2):

    # Shorten parameters
    Xtr_pca = X_train
    ytr_flat = y_train
    Xte_pca = X_test

    # Define optimals
    opts = {
        "random_state": None,
        "estimator": None,
        "n_estimators": 10,
        "max_samples": 1.0,
        "max_features": 1.0,
        "bootstrap": True,
        "bootstrap_features": False,
        "oob_score": False,
        "warm_start": False,
        "n_jobs": -1,
        "verbose": 0,
    }

    # Cyclically optimize hyperparameters
    ma_vec = []
    f1_vec = []
    for c in np.arange(cycles):
        opts, _, _ = _optimize_rs(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _, _ = _optimize_est(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _, _ = _optimize_nest(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _, _ = _optimize_maxs(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _, _ = _optimize_maxf(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _, _ = _optimize_boot(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _, _ = _optimize_bootf(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _, _ = _optimize_oob(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, ma, f1 = _optimize_warm(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        ma_vec.append(ma)
        f1_vec.append(f1)

    return opts, ma, f1, ma_vec, f1_vec
