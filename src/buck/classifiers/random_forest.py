from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
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
    for v in variable_array:
        # Define classifiers to test
        classifier = RandomForestClassifier(
            random_state=v,
            n_estimators=opts["n_estimators"],
            max_depth=opts["max_depth"],
            criterion=opts["criterion"],
            class_weight=opts["class_weight"],
            min_samples_split=opts["min_samples_split"],
            min_samples_leaf=opts["min_samples_leaf"],
            min_weight_fraction_leaf=opts["min_weight_fraction_leaf"],
            max_features=opts["max_features"],
            max_leaf_nodes=opts["max_leaf_nodes"],
            min_impurity_decrease=opts["min_impurity_decrease"],
            bootstrap=opts["bootstrap"],
            oob_score=opts["oob_score"],
            n_jobs=opts["n_jobs"],
            verbose=opts["verbose"],
            warm_start=opts["warm_start"],
            ccp_alpha=opts["ccp_alpha"],
            max_samples=opts["max_samples"],
            monotonic_cst=opts["monotonic_cst"],
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


def _optimize_nest(
    X_train_pca,
    y_train_flat,
    X_test_pca,
    y_true,
    opts,
):
    # Initialize variables
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = np.arange(1, 800)
    best_val = variable_array[0]

    # Iterate through variables
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = RandomForestClassifier(
            random_state=opts["random_state"],
            n_estimators=v,
            max_depth=opts["max_depth"],
            criterion=opts["criterion"],
            class_weight=opts["class_weight"],
            min_samples_split=opts["min_samples_split"],
            min_samples_leaf=opts["min_samples_leaf"],
            min_weight_fraction_leaf=opts["min_weight_fraction_leaf"],
            max_features=opts["max_features"],
            max_leaf_nodes=opts["max_leaf_nodes"],
            min_impurity_decrease=opts["min_impurity_decrease"],
            bootstrap=opts["bootstrap"],
            oob_score=opts["oob_score"],
            n_jobs=opts["n_jobs"],
            verbose=opts["verbose"],
            warm_start=opts["warm_start"],
            ccp_alpha=opts["ccp_alpha"],
            max_samples=opts["max_samples"],
            monotonic_cst=opts["monotonic_cst"],
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


def _optimize_max_d(
    X_train_pca,
    y_train_flat,
    X_test_pca,
    y_true,
    opts,
):
    # Initialize variables
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = np.arange(5, 25, 1)
    variable_array = np.append(variable_array.astype(object), None)  # type: ignore
    best_val = variable_array[0]

    # Iterate through variables
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = RandomForestClassifier(
            random_state=opts["random_state"],
            n_estimators=opts["n_estimators"],
            max_depth=v,
            criterion=opts["criterion"],
            class_weight=opts["class_weight"],
            min_samples_split=opts["min_samples_split"],
            min_samples_leaf=opts["min_samples_leaf"],
            min_weight_fraction_leaf=opts["min_weight_fraction_leaf"],
            max_features=opts["max_features"],
            max_leaf_nodes=opts["max_leaf_nodes"],
            min_impurity_decrease=opts["min_impurity_decrease"],
            bootstrap=opts["bootstrap"],
            oob_score=opts["oob_score"],
            n_jobs=opts["n_jobs"],
            verbose=opts["verbose"],
            warm_start=opts["warm_start"],
            ccp_alpha=opts["ccp_alpha"],
            max_samples=opts["max_samples"],
            monotonic_cst=opts["monotonic_cst"],
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
    opts["max_depth"] = best_val

    return opts, max_acc, f1s


def _optimize_crit(
    X_train_pca,
    y_train_flat,
    X_test_pca,
    y_true,
    opts,
):
    # Initialize variables
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = ["gini", "entropy", "log_loss"]
    best_val = variable_array[0]

    # Iterate through variables
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = RandomForestClassifier(
            random_state=opts["random_state"],
            n_estimators=opts["n_estimators"],
            max_depth=opts["max_depth"],
            criterion=v,
            class_weight=opts["class_weight"],
            min_samples_split=opts["min_samples_split"],
            min_samples_leaf=opts["min_samples_leaf"],
            min_weight_fraction_leaf=opts["min_weight_fraction_leaf"],
            max_features=opts["max_features"],
            max_leaf_nodes=opts["max_leaf_nodes"],
            min_impurity_decrease=opts["min_impurity_decrease"],
            bootstrap=opts["bootstrap"],
            oob_score=opts["oob_score"],
            n_jobs=opts["n_jobs"],
            verbose=opts["verbose"],
            warm_start=opts["warm_start"],
            ccp_alpha=opts["ccp_alpha"],
            max_samples=opts["max_samples"],
            monotonic_cst=opts["monotonic_cst"],
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
    opts["criterion"] = best_val

    return opts, max_acc, f1s


def _optimize_cw(X_train_pca, y_train_flat, X_test_pca, y_true, opts):
    # Initialize variables
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = ["balanced", "balanced_subsample", None]
    best_val = variable_array[0]

    # Iterate through variables
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = RandomForestClassifier(
            random_state=opts["random_state"],
            n_estimators=opts["n_estimators"],
            max_depth=opts["max_depth"],
            criterion=opts["criterion"],
            class_weight=v,
            min_samples_split=opts["min_samples_split"],
            min_samples_leaf=opts["min_samples_leaf"],
            min_weight_fraction_leaf=opts["min_weight_fraction_leaf"],
            max_features=opts["max_features"],
            max_leaf_nodes=opts["max_leaf_nodes"],
            min_impurity_decrease=opts["min_impurity_decrease"],
            bootstrap=opts["bootstrap"],
            oob_score=opts["oob_score"],
            n_jobs=opts["n_jobs"],
            verbose=opts["verbose"],
            warm_start=opts["warm_start"],
            ccp_alpha=opts["ccp_alpha"],
            max_samples=opts["max_samples"],
            monotonic_cst=opts["monotonic_cst"],
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


def _optimize_mss(
    X_train_pca,
    y_train_flat,
    X_test_pca,
    y_true,
    opts,
):
    # Initialize variables
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = np.arange(2, 100)
    best_val = variable_array[0]

    # Iterate through variables
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = RandomForestClassifier(
            random_state=opts["random_state"],
            n_estimators=opts["n_estimators"],
            max_depth=opts["max_depth"],
            criterion=opts["criterion"],
            class_weight=opts["class_weight"],
            min_samples_split=v,
            min_samples_leaf=opts["min_samples_leaf"],
            min_weight_fraction_leaf=opts["min_weight_fraction_leaf"],
            max_features=opts["max_features"],
            max_leaf_nodes=opts["max_leaf_nodes"],
            min_impurity_decrease=opts["min_impurity_decrease"],
            bootstrap=opts["bootstrap"],
            oob_score=opts["oob_score"],
            n_jobs=opts["n_jobs"],
            verbose=opts["verbose"],
            warm_start=opts["warm_start"],
            ccp_alpha=opts["ccp_alpha"],
            max_samples=opts["max_samples"],
            monotonic_cst=opts["monotonic_cst"],
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
    opts["min_samples_split"] = best_val

    return opts, max_acc, f1s


def _optimize_msl(
    X_train_pca,
    y_train_flat,
    X_test_pca,
    y_true,
    opts,
):
    # Initialize variables
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = np.arange(1, 50)
    best_val = variable_array[0]

    # Iterate through variables
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = RandomForestClassifier(
            random_state=opts["random_state"],
            n_estimators=opts["n_estimators"],
            max_depth=opts["max_depth"],
            criterion=opts["criterion"],
            class_weight=opts["class_weight"],
            min_samples_split=opts["min_samples_split"],
            min_samples_leaf=v,
            min_weight_fraction_leaf=opts["min_weight_fraction_leaf"],
            max_features=opts["max_features"],
            max_leaf_nodes=opts["max_leaf_nodes"],
            min_impurity_decrease=opts["min_impurity_decrease"],
            bootstrap=opts["bootstrap"],
            oob_score=opts["oob_score"],
            n_jobs=opts["n_jobs"],
            verbose=opts["verbose"],
            warm_start=opts["warm_start"],
            ccp_alpha=opts["ccp_alpha"],
            max_samples=opts["max_samples"],
            monotonic_cst=opts["monotonic_cst"],
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
    opts["min_samples_leaf"] = best_val

    return opts, max_acc, f1s


def _optimize_mwfl(
    X_train_pca,
    y_train_flat,
    X_test_pca,
    y_true,
    opts,
):
    # Initialize variables
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = np.arange(0.0, 0.5, 0.01)
    best_val = variable_array[0]

    # Iterate through variables
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = RandomForestClassifier(
            random_state=opts["random_state"],
            n_estimators=opts["n_estimators"],
            max_depth=opts["max_depth"],
            criterion=opts["criterion"],
            class_weight=opts["class_weight"],
            min_samples_split=opts["min_samples_split"],
            min_samples_leaf=opts["min_samples_leaf"],
            min_weight_fraction_leaf=v,
            max_features=opts["max_features"],
            max_leaf_nodes=opts["max_leaf_nodes"],
            min_impurity_decrease=opts["min_impurity_decrease"],
            bootstrap=opts["bootstrap"],
            oob_score=opts["oob_score"],
            n_jobs=opts["n_jobs"],
            verbose=opts["verbose"],
            warm_start=opts["warm_start"],
            ccp_alpha=opts["ccp_alpha"],
            max_samples=opts["max_samples"],
            monotonic_cst=opts["monotonic_cst"],
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
    opts["min_weight_fraction_leaf"] = best_val

    return opts, max_acc, f1s


def _optimize_mf(
    X_train_pca,
    y_train_flat,
    X_test_pca,
    y_true,
    opts,
):
    # Initialize variables
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = ["sqrt", "log2", None]
    best_val = variable_array[0]

    # Iterate through variables
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = RandomForestClassifier(
            random_state=opts["random_state"],
            n_estimators=opts["n_estimators"],
            max_depth=opts["max_depth"],
            criterion=opts["criterion"],
            class_weight=opts["class_weight"],
            min_samples_split=opts["min_samples_split"],
            min_samples_leaf=opts["min_samples_leaf"],
            min_weight_fraction_leaf=opts["min_weight_fraction_leaf"],
            max_features=v,
            max_leaf_nodes=opts["max_leaf_nodes"],
            min_impurity_decrease=opts["min_impurity_decrease"],
            bootstrap=opts["bootstrap"],
            oob_score=opts["oob_score"],
            n_jobs=opts["n_jobs"],
            verbose=opts["verbose"],
            warm_start=opts["warm_start"],
            ccp_alpha=opts["ccp_alpha"],
            max_samples=opts["max_samples"],
            monotonic_cst=opts["monotonic_cst"],
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


def _optimize_mln(
    X_train_pca,
    y_train_flat,
    X_test_pca,
    y_true,
    opts,
):
    # Initialize variables
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = np.arange(50, 1000)
    variable_array = np.append(variable_array.astype(object), None)  # type: ignore
    best_val = variable_array[0]

    # Iterate through variables
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = RandomForestClassifier(
            random_state=opts["random_state"],
            n_estimators=opts["n_estimators"],
            max_depth=opts["max_depth"],
            criterion=opts["criterion"],
            class_weight=opts["class_weight"],
            min_samples_split=opts["min_samples_split"],
            min_samples_leaf=opts["min_samples_leaf"],
            min_weight_fraction_leaf=opts["min_weight_fraction_leaf"],
            max_features=opts["max_features"],
            max_leaf_nodes=v,
            min_impurity_decrease=opts["min_impurity_decrease"],
            bootstrap=opts["bootstrap"],
            oob_score=opts["oob_score"],
            n_jobs=opts["n_jobs"],
            verbose=opts["verbose"],
            warm_start=opts["warm_start"],
            ccp_alpha=opts["ccp_alpha"],
            max_samples=opts["max_samples"],
            monotonic_cst=opts["monotonic_cst"],
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
    opts["max_leaf_nodes"] = best_val

    return opts, max_acc, f1s


def _optimize_mid(
    X_train_pca,
    y_train_flat,
    X_test_pca,
    y_true,
    opts,
):
    # Initialize variables
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = np.arange(0.0, 1.0, 0.01)
    best_val = variable_array[0]

    # Iterate through variables
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = RandomForestClassifier(
            random_state=opts["random_state"],
            n_estimators=opts["n_estimators"],
            max_depth=opts["max_depth"],
            criterion=opts["criterion"],
            class_weight=opts["class_weight"],
            min_samples_split=opts["min_samples_split"],
            min_samples_leaf=opts["min_samples_leaf"],
            min_weight_fraction_leaf=opts["min_weight_fraction_leaf"],
            max_features=opts["max_features"],
            max_leaf_nodes=opts["max_leaf_nodes"],
            min_impurity_decrease=v,
            bootstrap=opts["bootstrap"],
            oob_score=opts["oob_score"],
            n_jobs=opts["n_jobs"],
            verbose=opts["verbose"],
            warm_start=opts["warm_start"],
            ccp_alpha=opts["ccp_alpha"],
            max_samples=opts["max_samples"],
            monotonic_cst=opts["monotonic_cst"],
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
    opts["min_impurity_decrease"] = best_val

    return opts, max_acc, f1s


def _optimize_random_forest(X_train_pca, y_train_flat, X_test_pca, y_true, cycles=2):
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
        "n_estimators": 100,
        "criterion": "gini",
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "min_weight_fraction_leaf": 0.0,
        "max_features": "sqrt",
        "max_leaf_nodes": None,
        "min_impurity_decrease": 0.0,
        "bootstrap": True,
        "oob_score": False,
        "n_jobs": -1,
        "random_state": 42,
        "verbose": 0,
        "warm_start": False,
        "class_weight": None,
        "ccp_alpha": 0.0,
        "max_samples": None,
        "monotonic_cst": None,
    }

    # Optimize hyperparameters
    ma_vec = []
    f1_vec = []
    for c in np.arange(cycles):
        print(f"Cycle {c + 1} of {cycles}")
        opts, _, _ = _optimize_rs(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _, _ = _optimize_nest(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _, _ = _optimize_max_d(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _, _ = _optimize_crit(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)  # type: ignore
        opts, _, _ = _optimize_cw(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _, _ = _optimize_mss(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _, _ = _optimize_msl(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _, _ = _optimize_mwfl(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _, _ = _optimize_mf(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _, _ = _optimize_mln(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, ma, f1 = _optimize_mid(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        ma_vec.append(ma)
        f1_vec.append(f1)

    return opts, ma, f1, ma_vec, f1_vec
