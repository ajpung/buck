from typing import Any

import numpy as np
from sklearn.ensemble import (
    VotingClassifier,
    BaggingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
)
from sklearn.metrics import accuracy_score, f1_score


def _optimize_voting_type(X_train, y_train, X_test, y_true, opts):
    """Optimize voting type (hard vs soft)"""
    max_acc = -np.inf
    variable_array = ["hard", "soft"]
    best_val = variable_array[0]

    for v in variable_array:
        # Create estimators
        bagging = BaggingClassifier(**opts["bagging"])
        rf = RandomForestClassifier(**opts["rf"])
        et = ExtraTreesClassifier(**opts["et"])

        classifier = VotingClassifier(
            estimators=[("bagging", bagging), ("rf", rf), ("et", et)],
            voting=v,
            n_jobs=opts["voting"]["n_jobs"],
            flatten_transform=opts["voting"]["flatten_transform"],
            verbose=opts["voting"]["verbose"],
        )

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        if accuracy >= max_acc:
            max_acc = accuracy
            f1s = f1
            best_val = v

    opts["voting"]["voting"] = best_val
    return opts, max_acc, f1s


def _optimize_bagging_n_estimators(X_train, y_train, X_test, y_true, opts):
    """Optimize BaggingClassifier n_estimators"""
    max_acc = -np.inf
    variable_array = np.arange(10, 200, 10)
    best_val = variable_array[0]

    for v in variable_array:
        opts["bagging"]["n_estimators"] = v

        bagging = BaggingClassifier(**opts["bagging"])
        rf = RandomForestClassifier(**opts["rf"])
        et = ExtraTreesClassifier(**opts["et"])

        classifier = VotingClassifier(
            estimators=[("bagging", bagging), ("rf", rf), ("et", et)], **opts["voting"]
        )

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        if accuracy >= max_acc:
            max_acc = accuracy
            f1s = f1
            best_val = v

    opts["bagging"]["n_estimators"] = best_val
    return opts, max_acc, f1s


def _optimize_bagging_max_samples(X_train, y_train, X_test, y_true, opts):
    """Optimize BaggingClassifier max_samples"""
    max_acc = -np.inf
    variable_array = np.arange(0.5, 1.1, 0.1)
    best_val = variable_array[0]

    for v in variable_array:
        opts["bagging"]["max_samples"] = v

        bagging = BaggingClassifier(**opts["bagging"])
        rf = RandomForestClassifier(**opts["rf"])
        et = ExtraTreesClassifier(**opts["et"])

        classifier = VotingClassifier(
            estimators=[("bagging", bagging), ("rf", rf), ("et", et)], **opts["voting"]
        )

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        if accuracy >= max_acc:
            max_acc = accuracy
            f1s = f1
            best_val = v

    opts["bagging"]["max_samples"] = best_val
    return opts, max_acc, f1s


def _optimize_rf_n_estimators(X_train, y_train, X_test, y_true, opts):
    """Optimize RandomForest n_estimators"""
    max_acc = -np.inf
    variable_array = np.arange(50, 300, 25)
    best_val = variable_array[0]

    for v in variable_array:
        opts["rf"]["n_estimators"] = v

        bagging = BaggingClassifier(**opts["bagging"])
        rf = RandomForestClassifier(**opts["rf"])
        et = ExtraTreesClassifier(**opts["et"])

        classifier = VotingClassifier(
            estimators=[("bagging", bagging), ("rf", rf), ("et", et)], **opts["voting"]
        )

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        if accuracy >= max_acc:
            max_acc = accuracy
            f1s = f1
            best_val = v

    opts["rf"]["n_estimators"] = best_val
    return opts, max_acc, f1s


def _optimize_rf_max_depth(X_train, y_train, X_test, y_true, opts):
    """Optimize RandomForest max_depth"""
    max_acc = -np.inf
    variable_array = np.arange(5, 25, 2)
    variable_array = np.append(variable_array.astype(object), None)
    best_val = variable_array[0]

    for v in variable_array:
        opts["rf"]["max_depth"] = v

        bagging = BaggingClassifier(**opts["bagging"])
        rf = RandomForestClassifier(**opts["rf"])
        et = ExtraTreesClassifier(**opts["et"])

        classifier = VotingClassifier(
            estimators=[("bagging", bagging), ("rf", rf), ("et", et)], **opts["voting"]
        )

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        if accuracy >= max_acc:
            max_acc = accuracy
            f1s = f1
            best_val = v

    opts["rf"]["max_depth"] = best_val
    return opts, max_acc, f1s


def _optimize_et_n_estimators(X_train, y_train, X_test, y_true, opts):
    """Optimize ExtraTrees n_estimators"""
    max_acc = -np.inf
    variable_array = np.arange(50, 300, 25)
    best_val = variable_array[0]

    for v in variable_array:
        opts["et"]["n_estimators"] = v

        bagging = BaggingClassifier(**opts["bagging"])
        rf = RandomForestClassifier(**opts["rf"])
        et = ExtraTreesClassifier(**opts["et"])

        classifier = VotingClassifier(
            estimators=[("bagging", bagging), ("rf", rf), ("et", et)], **opts["voting"]
        )

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        if accuracy >= max_acc:
            max_acc = accuracy
            f1s = f1
            best_val = v

    opts["et"]["n_estimators"] = best_val
    return opts, max_acc, f1s


def _optimize_et_max_depth(X_train, y_train, X_test, y_true, opts):
    """Optimize ExtraTrees max_depth"""
    max_acc = -np.inf
    variable_array = np.arange(5, 25, 2)
    variable_array = np.append(variable_array.astype(object), None)
    best_val = variable_array[0]

    for v in variable_array:
        opts["et"]["max_depth"] = v

        bagging = BaggingClassifier(**opts["bagging"])
        rf = RandomForestClassifier(**opts["rf"])
        et = ExtraTreesClassifier(**opts["et"])

        classifier = VotingClassifier(
            estimators=[("bagging", bagging), ("rf", rf), ("et", et)], **opts["voting"]
        )

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        if accuracy >= max_acc:
            max_acc = accuracy
            f1s = f1
            best_val = v

    opts["et"]["max_depth"] = best_val
    return opts, max_acc, f1s


def _optimize_max_features(X_train, y_train, X_test, y_true, opts):
    """Optimize max_features for both RF and ET"""
    max_acc = -np.inf
    variable_array = ["sqrt", "log2", None]
    best_val = variable_array[0]

    for v in variable_array:
        opts["rf"]["max_features"] = v
        opts["et"]["max_features"] = v

        bagging = BaggingClassifier(**opts["bagging"])
        rf = RandomForestClassifier(**opts["rf"])
        et = ExtraTreesClassifier(**opts["et"])

        classifier = VotingClassifier(
            estimators=[("bagging", bagging), ("rf", rf), ("et", et)], **opts["voting"]
        )

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        if accuracy >= max_acc:
            max_acc = accuracy
            f1s = f1
            best_val = v

    opts["rf"]["max_features"] = best_val
    opts["et"]["max_features"] = best_val
    return opts, max_acc, f1s


def _optimize_min_samples_split(X_train, y_train, X_test, y_true, opts):
    """Optimize min_samples_split for both RF and ET"""
    max_acc = -np.inf
    variable_array = np.arange(2, 20, 2)
    best_val = variable_array[0]

    for v in variable_array:
        opts["rf"]["min_samples_split"] = v
        opts["et"]["min_samples_split"] = v

        bagging = BaggingClassifier(**opts["bagging"])
        rf = RandomForestClassifier(**opts["rf"])
        et = ExtraTreesClassifier(**opts["et"])

        classifier = VotingClassifier(
            estimators=[("bagging", bagging), ("rf", rf), ("et", et)], **opts["voting"]
        )

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        if accuracy >= max_acc:
            max_acc = accuracy
            f1s = f1
            best_val = v

    opts["rf"]["min_samples_split"] = best_val
    opts["et"]["min_samples_split"] = best_val
    return opts, max_acc, f1s


def _optimize_voting_classifier(X_train, y_train, X_test, y_true, cycles=2):
    """
    Optimizes the hyperparameters for a VotingClassifier with Bagging, RandomForest, and ExtraTrees.
    """
    # Initialize default parameters
    opts = {
        "voting": {
            "voting": "soft",
            "n_jobs": -1,
            "flatten_transform": True,
            "verbose": 0,
        },
        "bagging": {
            "n_estimators": 100,
            "max_samples": 1.0,
            "max_features": 1.0,
            "bootstrap": True,
            "bootstrap_features": False,
            "oob_score": False,
            "warm_start": False,
            "n_jobs": -1,
            "random_state": 42,
            "verbose": 0,
        },
        "rf": {
            "n_estimators": 100,
            "criterion": "gini",
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "bootstrap": True,
            "oob_score": False,
            "n_jobs": -1,
            "random_state": 42,
            "verbose": 0,
        },
        "et": {
            "n_estimators": 100,
            "criterion": "gini",
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "bootstrap": False,
            "oob_score": False,
            "n_jobs": -1,
            "random_state": 42,
            "verbose": 0,
        },
    }

    # Optimize hyperparameters
    ma_vec = []
    f1_vec = []

    for c in np.arange(cycles):
        print(f"Cycle {c + 1} of {cycles}")

        opts, _, _ = _optimize_voting_type(X_train, y_train, X_test, y_true, opts)
        opts, _, _ = _optimize_bagging_n_estimators(
            X_train, y_train, X_test, y_true, opts
        )
        opts, _, _ = _optimize_bagging_max_samples(
            X_train, y_train, X_test, y_true, opts
        )
        opts, _, _ = _optimize_rf_n_estimators(X_train, y_train, X_test, y_true, opts)
        opts, _, _ = _optimize_rf_max_depth(X_train, y_train, X_test, y_true, opts)
        opts, _, _ = _optimize_et_n_estimators(X_train, y_train, X_test, y_true, opts)
        opts, _, _ = _optimize_et_max_depth(X_train, y_train, X_test, y_true, opts)
        opts, _, _ = _optimize_max_features(X_train, y_train, X_test, y_true, opts)
        opts, ma, f1 = _optimize_min_samples_split(
            X_train, y_train, X_test, y_true, opts
        )

        ma_vec.append(ma)
        f1_vec.append(f1)

    return opts, ma, f1, ma_vec, f1_vec
