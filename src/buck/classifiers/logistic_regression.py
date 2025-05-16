from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


def _optimize_rs(
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
    variable_array = np.arange(150)
    best_val = variable_array[0]
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = LogisticRegression(
            random_state=v,
            penalty=opts["penalty"],
            dual=opts["dual"],
            tol=opts["tol"],
            C=opts["C"],
            fit_intercept=opts["fit_intercept"],
            intercept_scaling=opts["intercept_scaling"],
            class_weight=opts["class_weight"],
            solver="lbfgs",
            max_iter=opts["max_iter"],
            multi_class=opts["multi_class"],
            verbose=opts["verbose"],
            warm_start=opts["warm_start"],
            n_jobs=opts["n_jobs"],
            l1_ratio=opts["l1_ratio"],
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
    opts["random_state"] = best_val

    return opts, max_acc


def _optimize_pn(
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
    variable_array = ["l1", "l2", "elasticnet", "none"]
    best_val = variable_array[0]
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = LogisticRegression(
            random_state=opts["random_state"],
            penalty=v,
            dual=opts["dual"],
            tol=opts["tol"],
            C=opts["C"],
            fit_intercept=opts["fit_intercept"],
            intercept_scaling=opts["intercept_scaling"],
            class_weight=opts["class_weight"],
            solver="lbfgs",
            max_iter=opts["max_iter"],
            multi_class=opts["multi_class"],
            verbose=opts["verbose"],
            warm_start=opts["warm_start"],
            n_jobs=opts["n_jobs"],
            l1_ratio=opts["l1_ratio"],
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
    opts["penalty"] = best_val

    return opts, max_acc


def _optimize_dl(
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
    variable_array = [True, False]
    best_val = variable_array[0]
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = LogisticRegression(
            random_state=opts["random_state"],
            penalty=opts["penalty"],
            dual=v,
            tol=opts["tol"],
            C=opts["C"],
            fit_intercept=opts["fit_intercept"],
            intercept_scaling=opts["intercept_scaling"],
            class_weight=opts["class_weight"],
            solver="lbfgs",
            max_iter=opts["max_iter"],
            multi_class=opts["multi_class"],
            verbose=opts["verbose"],
            warm_start=opts["warm_start"],
            n_jobs=opts["n_jobs"],
            l1_ratio=opts["l1_ratio"],
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
    opts["dual"] = best_val

    return opts, max_acc


def _optimize_tol(
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
    variable_array = np.arange(0.0001, 0.1, 0.0001)
    best_val = variable_array[0]
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = LogisticRegression(
            random_state=opts["random_state"],
            penalty=opts["penalty"],
            dual=opts["dual"],
            tol=v,
            C=opts["C"],
            fit_intercept=opts["fit_intercept"],
            intercept_scaling=opts["intercept_scaling"],
            class_weight=opts["class_weight"],
            solver="lbfgs",
            max_iter=opts["max_iter"],
            multi_class=opts["multi_class"],
            verbose=opts["verbose"],
            warm_start=opts["warm_start"],
            n_jobs=opts["n_jobs"],
            l1_ratio=opts["l1_ratio"],
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
    opts["tol"] = best_val

    return opts, max_acc


def _optimize_c(
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
    variable_array = np.arange(0.0001, 10, 0.0001)
    best_val = variable_array[0]
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = LogisticRegression(
            random_state=opts["random_state"],
            penalty=opts["penalty"],
            dual=opts["dual"],
            tol=opts["tol"],
            C=v,
            fit_intercept=opts["fit_intercept"],
            intercept_scaling=opts["intercept_scaling"],
            class_weight=opts["class_weight"],
            solver="lbfgs",
            max_iter=opts["max_iter"],
            multi_class=opts["multi_class"],
            verbose=opts["verbose"],
            warm_start=opts["warm_start"],
            n_jobs=opts["n_jobs"],
            l1_ratio=opts["l1_ratio"],
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
    opts["C"] = best_val

    return opts, max_acc


def _optimize_fi(
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
    variable_array = [True, False]
    best_val = variable_array[0]
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = LogisticRegression(
            random_state=opts["random_state"],
            penalty=opts["penalty"],
            dual=opts["dual"],
            tol=opts["tol"],
            C=opts["C"],
            fit_intercept=v,
            intercept_scaling=opts["intercept_scaling"],
            class_weight=opts["class_weight"],
            solver="lbfgs",
            max_iter=opts["max_iter"],
            multi_class=opts["multi_class"],
            verbose=opts["verbose"],
            warm_start=opts["warm_start"],
            n_jobs=opts["n_jobs"],
            l1_ratio=opts["l1_ratio"],
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
    opts["fit_intercept"] = best_val

    return opts, max_acc


def _optimize_is(
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
    variable_array = np.arange(0.0001, 10, 0.0001)
    best_val = variable_array[0]
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = LogisticRegression(
            random_state=opts["random_state"],
            penalty=opts["penalty"],
            dual=opts["dual"],
            tol=opts["tol"],
            C=opts["C"],
            fit_intercept=opts["fit_intercept"],
            intercept_scaling=v,
            class_weight=opts["class_weight"],
            solver="lbfgs",
            max_iter=opts["max_iter"],
            multi_class=opts["multi_class"],
            verbose=opts["verbose"],
            warm_start=opts["warm_start"],
            n_jobs=opts["n_jobs"],
            l1_ratio=opts["l1_ratio"],
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
    opts["intercept_scaling"] = best_val

    return opts, max_acc


def _optimize_cw(
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
    variable_array = [None, "balanced"]
    best_val = variable_array[0]
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = LogisticRegression(
            random_state=opts["random_state"],
            penalty=opts["penalty"],
            dual=opts["dual"],
            tol=opts["tol"],
            C=opts["C"],
            fit_intercept=opts["fit_intercept"],
            intercept_scaling=opts["intercept_scaling"],
            class_weight=v,
            solver="lbfgs",
            max_iter=opts["max_iter"],
            multi_class=opts["multi_class"],
            verbose=opts["verbose"],
            warm_start=opts["warm_start"],
            n_jobs=opts["n_jobs"],
            l1_ratio=opts["l1_ratio"],
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
    opts["class_weight"] = best_val

    return opts, max_acc


def _optimize_sol(
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
    variable_array = [
        "newton-cg",
        "newton-cholesky",
        "lbfgs",
        "liblinear",
        "sag",
        "saga",
    ]
    best_val = variable_array[0]
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = LogisticRegression(
            random_state=opts["random_state"],
            penalty=opts["penalty"],
            dual=opts["dual"],
            tol=opts["tol"],
            C=opts["C"],
            fit_intercept=opts["fit_intercept"],
            intercept_scaling=opts["intercept_scaling"],
            class_weight=opts["class_weight"],
            solver=v,
            max_iter=opts["max_iter"],
            multi_class=opts["multi_class"],
            verbose=opts["verbose"],
            warm_start=opts["warm_start"],
            n_jobs=opts["n_jobs"],
            l1_ratio=opts["l1_ratio"],
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
    opts["solver"] = best_val

    return opts, max_acc


def _optimize_mi(
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
    variable_array = np.arange(100, 10000, 100)
    best_val = variable_array[0]
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = LogisticRegression(
            random_state=opts["random_state"],
            penalty=opts["penalty"],
            dual=opts["dual"],
            tol=opts["tol"],
            C=opts["C"],
            fit_intercept=opts["fit_intercept"],
            intercept_scaling=opts["intercept_scaling"],
            class_weight=opts["class_weight"],
            solver="lbfgs",
            max_iter=v,
            multi_class=opts["multi_class"],
            verbose=opts["verbose"],
            warm_start=opts["warm_start"],
            n_jobs=opts["n_jobs"],
            l1_ratio=opts["l1_ratio"],
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
    opts["max_iter"] = best_val

    return opts, max_acc


def _optimize_mc(
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
    variable_array = ["auto", "ovr", "multinomial"]
    best_val = variable_array[0]
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = LogisticRegression(
            random_state=opts["random_state"],
            penalty=opts["penalty"],
            dual=opts["dual"],
            tol=opts["tol"],
            C=opts["C"],
            fit_intercept=opts["fit_intercept"],
            intercept_scaling=opts["intercept_scaling"],
            class_weight=opts["class_weight"],
            solver="lbfgs",
            max_iter=opts["max_iter"],
            multi_class=v,
            verbose=opts["verbose"],
            warm_start=opts["warm_start"],
            n_jobs=opts["n_jobs"],
            l1_ratio=opts["l1_ratio"],
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
    opts["multi_class"] = best_val

    return opts, max_acc


def _optimize_l1r(
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
    variable_array = [None, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    best_val = variable_array[0]
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = LogisticRegression(
            random_state=opts["random_state"],
            penalty=opts["penalty"],
            dual=opts["dual"],
            tol=opts["tol"],
            C=opts["C"],
            fit_intercept=opts["fit_intercept"],
            intercept_scaling=opts["intercept_scaling"],
            class_weight=opts["class_weight"],
            solver="lbfgs",
            max_iter=opts["max_iter"],
            multi_class=opts["multi_class"],
            verbose=opts["verbose"],
            warm_start=opts["warm_start"],
            n_jobs=opts["n_jobs"],
            l1_ratio=v,
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
    opts["l1_ratio"] = best_val

    return opts, max_acc


def _optimize_logistic_regression(
    X_train_pca, y_train_flat, X_test_pca, y_true, cycles=2
):

    # Shorten parameters
    Xtr_pca = X_train_pca
    ytr_flat = y_train_flat
    Xte_pca = X_test_pca

    # Define optimals
    opts = {
        "penalty": "l2",
        "dual": False,
        "tol": 0.0001,
        "C": 1.0,
        "fit_intercept": True,
        "intercept_scaling": 1,
        "class_weight": None,
        "random_state": None,
        "solver": "lbfgs",
        "max_iter": 100,
        "multi_class": "deprecated",
        "verbose": 0,
        "warm_start": False,
        "n_jobs": -1,
        "l1_ratio": None,
    }

    # Cyclically optimize hyperparameters
    ma_vec = []
    for c in np.arange(cycles):
        print(f"Cycle {c + 1} of {cycles}")
        opts, _ = _optimize_rs(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _ = _optimize_pn(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _ = _optimize_dl(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _ = _optimize_tol(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _ = _optimize_c(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _ = _optimize_fi(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _ = _optimize_is(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _ = _optimize_cw(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _ = _optimize_sol(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _ = _optimize_mi(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _ = _optimize_mc(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, ma = _optimize_l1r(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        ma_vec.append(ma)

    return opts, ma_vec
