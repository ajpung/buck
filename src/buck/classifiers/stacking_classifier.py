from typing import Any

import numpy as np
from sklearn.ensemble import (
    StackingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score


def _optimize_cv(X_train, y_train, X_test, y_true, opts):
    """Optimize cross-validation folds"""
    max_acc = -np.inf
    variable_array = np.arange(3, 11)  # 3 to 10 folds
    best_val = variable_array[0]

    for v in variable_array:
        opts["stacking"]["cv"] = v

        # Create base estimators
        rf = RandomForestClassifier(**opts["rf"])
        et = ExtraTreesClassifier(**opts["et"])
        sgd = SGDClassifier(**opts["sgd"])
        svc = SVC(**opts["svc"])

        # Create meta estimator
        meta = LogisticRegression(**opts["meta"])

        classifier = StackingClassifier(
            estimators=[("rf", rf), ("et", et), ("sgd", sgd), ("svc", svc)],
            final_estimator=meta,
            cv=v,
            stack_method=opts["stacking"]["stack_method"],
            n_jobs=opts["stacking"]["n_jobs"],
            passthrough=opts["stacking"]["passthrough"],
            verbose=opts["stacking"]["verbose"],
        )

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        if accuracy >= max_acc:
            max_acc = accuracy
            f1s = f1
            best_val = v

    opts["stacking"]["cv"] = best_val
    return opts, max_acc, f1s


def _optimize_stack_method(X_train, y_train, X_test, y_true, opts):
    """Optimize stacking method"""
    max_acc = -np.inf
    variable_array = ["auto", "predict_proba", "decision_function", "predict"]
    best_val = variable_array[0]

    for v in variable_array:
        try:
            opts["stacking"]["stack_method"] = v

            # Create base estimators
            rf = RandomForestClassifier(**opts["rf"])
            et = ExtraTreesClassifier(**opts["et"])
            sgd = SGDClassifier(**opts["sgd"])
            svc = SVC(**opts["svc"])

            # Create meta estimator
            meta = LogisticRegression(**opts["meta"])

            classifier = StackingClassifier(
                estimators=[("rf", rf), ("et", et), ("sgd", sgd), ("svc", svc)],
                final_estimator=meta,
                cv=opts["stacking"]["cv"],
                stack_method=v,
                n_jobs=opts["stacking"]["n_jobs"],
                passthrough=opts["stacking"]["passthrough"],
                verbose=opts["stacking"]["verbose"],
            )

            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

            if accuracy >= max_acc:
                max_acc = accuracy
                f1s = f1
                best_val = v
        except:
            # Skip invalid combinations
            continue

    opts["stacking"]["stack_method"] = best_val
    return opts, max_acc, f1s


def _optimize_passthrough(X_train, y_train, X_test, y_true, opts):
    """Optimize passthrough parameter"""
    max_acc = -np.inf
    variable_array = [True, False]
    best_val = variable_array[0]

    for v in variable_array:
        opts["stacking"]["passthrough"] = v

        # Create base estimators
        rf = RandomForestClassifier(**opts["rf"])
        et = ExtraTreesClassifier(**opts["et"])
        sgd = SGDClassifier(**opts["sgd"])
        svc = SVC(**opts["svc"])

        # Create meta estimator
        meta = LogisticRegression(**opts["meta"])

        classifier = StackingClassifier(
            estimators=[("rf", rf), ("et", et), ("sgd", sgd), ("svc", svc)],
            final_estimator=meta,
            cv=opts["stacking"]["cv"],
            stack_method=opts["stacking"]["stack_method"],
            n_jobs=opts["stacking"]["n_jobs"],
            passthrough=v,
            verbose=opts["stacking"]["verbose"],
        )

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        if accuracy >= max_acc:
            max_acc = accuracy
            f1s = f1
            best_val = v

    opts["stacking"]["passthrough"] = best_val
    return opts, max_acc, f1s


def _optimize_rf_n_estimators(X_train, y_train, X_test, y_true, opts):
    """Optimize RandomForest n_estimators"""
    max_acc = -np.inf
    variable_array = np.arange(50, 200, 25)
    best_val = variable_array[0]

    for v in variable_array:
        opts["rf"]["n_estimators"] = v

        # Create base estimators
        rf = RandomForestClassifier(**opts["rf"])
        et = ExtraTreesClassifier(**opts["et"])
        sgd = SGDClassifier(**opts["sgd"])
        svc = SVC(**opts["svc"])

        # Create meta estimator
        meta = LogisticRegression(**opts["meta"])

        classifier = StackingClassifier(
            estimators=[("rf", rf), ("et", et), ("sgd", sgd), ("svc", svc)],
            final_estimator=meta,
            **opts["stacking"],
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


def _optimize_et_n_estimators(X_train, y_train, X_test, y_true, opts):
    """Optimize ExtraTrees n_estimators"""
    max_acc = -np.inf
    variable_array = np.arange(50, 200, 25)
    best_val = variable_array[0]

    for v in variable_array:
        opts["et"]["n_estimators"] = v

        # Create base estimators
        rf = RandomForestClassifier(**opts["rf"])
        et = ExtraTreesClassifier(**opts["et"])
        sgd = SGDClassifier(**opts["sgd"])
        svc = SVC(**opts["svc"])

        # Create meta estimator
        meta = LogisticRegression(**opts["meta"])

        classifier = StackingClassifier(
            estimators=[("rf", rf), ("et", et), ("sgd", sgd), ("svc", svc)],
            final_estimator=meta,
            **opts["stacking"],
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


def _optimize_sgd_alpha(X_train, y_train, X_test, y_true, opts):
    """Optimize SGD alpha"""
    max_acc = -np.inf
    variable_array = np.logspace(-5, 0, 8)
    best_val = variable_array[0]

    for v in variable_array:
        opts["sgd"]["alpha"] = v

        # Create base estimators
        rf = RandomForestClassifier(**opts["rf"])
        et = ExtraTreesClassifier(**opts["et"])
        sgd = SGDClassifier(**opts["sgd"])
        svc = SVC(**opts["svc"])

        # Create meta estimator
        meta = LogisticRegression(**opts["meta"])

        classifier = StackingClassifier(
            estimators=[("rf", rf), ("et", et), ("sgd", sgd), ("svc", svc)],
            final_estimator=meta,
            **opts["stacking"],
        )

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        if accuracy >= max_acc:
            max_acc = accuracy
            f1s = f1
            best_val = v

    opts["sgd"]["alpha"] = best_val
    return opts, max_acc, f1s


def _optimize_svc_C(X_train, y_train, X_test, y_true, opts):
    """Optimize SVC C parameter"""
    max_acc = -np.inf
    variable_array = np.logspace(-2, 2, 8)
    best_val = variable_array[0]

    for v in variable_array:
        opts["svc"]["C"] = v

        # Create base estimators
        rf = RandomForestClassifier(**opts["rf"])
        et = ExtraTreesClassifier(**opts["et"])
        sgd = SGDClassifier(**opts["sgd"])
        svc = SVC(**opts["svc"])

        # Create meta estimator
        meta = LogisticRegression(**opts["meta"])

        classifier = StackingClassifier(
            estimators=[("rf", rf), ("et", et), ("sgd", sgd), ("svc", svc)],
            final_estimator=meta,
            **opts["stacking"],
        )

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        if accuracy >= max_acc:
            max_acc = accuracy
            f1s = f1
            best_val = v

    opts["svc"]["C"] = best_val
    return opts, max_acc, f1s


def _optimize_meta_C(X_train, y_train, X_test, y_true, opts):
    """Optimize meta-learner (LogisticRegression) C parameter"""
    max_acc = -np.inf
    variable_array = np.logspace(-2, 2, 8)
    best_val = variable_array[0]

    for v in variable_array:
        opts["meta"]["C"] = v

        # Create base estimators
        rf = RandomForestClassifier(**opts["rf"])
        et = ExtraTreesClassifier(**opts["et"])
        sgd = SGDClassifier(**opts["sgd"])
        svc = SVC(**opts["svc"])

        # Create meta estimator
        meta = LogisticRegression(**opts["meta"])

        classifier = StackingClassifier(
            estimators=[("rf", rf), ("et", et), ("sgd", sgd), ("svc", svc)],
            final_estimator=meta,
            **opts["stacking"],
        )

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        if accuracy >= max_acc:
            max_acc = accuracy
            f1s = f1
            best_val = v

    opts["meta"]["C"] = best_val
    return opts, max_acc, f1s


def _optimize_stacking_classifier(X_train, y_train, X_test, y_true, cycles=2):
    """
    Optimizes the hyperparameters for a StackingClassifier with RF, ET, SGD, SVC base estimators
    and LogisticRegression as meta-learner.
    """

    # Initialize default parameters
    opts = {
        "stacking": {
            "cv": 5,
            "stack_method": "auto",
            "n_jobs": -1,
            "passthrough": False,
            "verbose": 0,
        },
        "rf": {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "bootstrap": True,
            "random_state": 42,
            "n_jobs": -1,
        },
        "et": {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "bootstrap": False,
            "random_state": 42,
            "n_jobs": -1,
        },
        "sgd": {
            "loss": "hinge",
            "alpha": 0.0001,
            "max_iter": 1000,
            "tol": 1e-3,
            "random_state": 42,
            "n_jobs": -1,
        },
        "svc": {
            "C": 1.0,
            "kernel": "rbf",
            "probability": True,  # Required for predict_proba
            "random_state": 42,
        },
        "meta": {
            "C": 1.0,
            "max_iter": 1000,
            "random_state": 42,
            "n_jobs": -1,
        },
    }

    # Optimize hyperparameters
    ma_vec = []
    f1_vec = []

    for c in np.arange(cycles):
        print(f"Cycle {c + 1} of {cycles}")

        opts, _, _ = _optimize_cv(X_train, y_train, X_test, y_true, opts)
        opts, _, _ = _optimize_stack_method(X_train, y_train, X_test, y_true, opts)
        opts, _, _ = _optimize_passthrough(X_train, y_train, X_test, y_true, opts)
        opts, _, _ = _optimize_rf_n_estimators(X_train, y_train, X_test, y_true, opts)
        opts, _, _ = _optimize_et_n_estimators(X_train, y_train, X_test, y_true, opts)
        opts, _, _ = _optimize_sgd_alpha(X_train, y_train, X_test, y_true, opts)
        opts, _, _ = _optimize_svc_C(X_train, y_train, X_test, y_true, opts)
        opts, ma, f1 = _optimize_meta_C(X_train, y_train, X_test, y_true, opts)

        ma_vec.append(ma)
        f1_vec.append(f1)

    return opts, ma, f1, ma_vec, f1_vec
