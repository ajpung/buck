from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier


# ----------------- RANDOM STATE -----------------
def _optimize_rs(
    X_train_pca, y_train_flat, X_test_pca, y_true, opts
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
        classifier = DecisionTreeClassifier(
            random_state=v,
            criterion=opts["criterion"],
            splitter=opts["splitter"],
            max_depth=opts["max_depth"],
            min_samples_split=opts["min_samples_split"],
            min_samples_leaf=opts["min_samples_leaf"],
            min_weight_fraction_leaf=opts["min_weight_fraction_leaf"],
            max_features=opts["max_features"],
            max_leaf_nodes=opts["max_leaf_nodes"],
            min_impurity_decrease=opts["min_impurity_decrease"],
            class_weight=opts["class_weight"],
            ccp_alpha=opts["ccp_alpha"],
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
            best_val = v

    # Store best value
    opts["random_state"] = best_val

    return opts, max_acc


def _optimize_ct(
    X_train_pca, y_train_flat, X_test_pca, y_true, opts
) -> tuple[Any, float]:
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = ["gini", "entropy", "log_loss"]
    best_val = variable_array[0]
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = DecisionTreeClassifier(
            random_state=opts["random_state"],
            criterion=v,
            splitter=opts["splitter"],
            max_depth=opts["max_depth"],
            min_samples_split=opts["min_samples_split"],
            min_samples_leaf=opts["min_samples_leaf"],
            min_weight_fraction_leaf=opts["min_weight_fraction_leaf"],
            max_features=opts["max_features"],
            max_leaf_nodes=opts["max_leaf_nodes"],
            min_impurity_decrease=opts["min_impurity_decrease"],
            class_weight=opts["class_weight"],
            ccp_alpha=opts["ccp_alpha"],
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
            best_val = v

    # Store best value
    opts["criterion"] = best_val

    return opts, max_acc


def _optimize_sp(
    X_train_pca, y_train_flat, X_test_pca, y_true, opts
) -> tuple[Any, float]:
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = ["best", "random"]
    best_val = variable_array[0]
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = DecisionTreeClassifier(
            random_state=opts["random_state"],
            criterion=opts["criterion"],
            splitter=v,
            max_depth=opts["max_depth"],
            min_samples_split=opts["min_samples_split"],
            min_samples_leaf=opts["min_samples_leaf"],
            min_weight_fraction_leaf=opts["min_weight_fraction_leaf"],
            max_features=opts["max_features"],
            max_leaf_nodes=opts["max_leaf_nodes"],
            min_impurity_decrease=opts["min_impurity_decrease"],
            class_weight=opts["class_weight"],
            ccp_alpha=opts["ccp_alpha"],
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
            best_val = v

    # Store best value
    opts["splitter"] = best_val

    return opts, max_acc


def _optimize_md(
    X_train_pca, y_train_flat, X_test_pca, y_true, opts
) -> tuple[Any, float]:
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = np.arange(1, 20)
    variable_array = np.append(variable_array.astype(object), None)  # type: ignore
    best_val = variable_array[0]
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = DecisionTreeClassifier(
            random_state=opts["random_state"],
            criterion=opts["criterion"],
            splitter=opts["splitter"],
            max_depth=v,
            min_samples_split=opts["min_samples_split"],
            min_samples_leaf=opts["min_samples_leaf"],
            min_weight_fraction_leaf=opts["min_weight_fraction_leaf"],
            max_features=opts["max_features"],
            max_leaf_nodes=opts["max_leaf_nodes"],
            min_impurity_decrease=opts["min_impurity_decrease"],
            class_weight=opts["class_weight"],
            ccp_alpha=opts["ccp_alpha"],
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
            best_val = v

    # Store best value
    opts["max_depth"] = best_val

    return opts, max_acc


def _optimize_mss(
    X_train_pca, y_train_flat, X_test_pca, y_true, opts
) -> tuple[Any, float]:
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = np.arange(2, 20)
    best_val = variable_array[0]
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = DecisionTreeClassifier(
            random_state=opts["random_state"],
            criterion=opts["criterion"],
            splitter=opts["splitter"],
            max_depth=opts["max_depth"],
            min_samples_split=v,
            min_samples_leaf=opts["min_samples_leaf"],
            min_weight_fraction_leaf=opts["min_weight_fraction_leaf"],
            max_features=opts["max_features"],
            max_leaf_nodes=opts["max_leaf_nodes"],
            min_impurity_decrease=opts["min_impurity_decrease"],
            class_weight=opts["class_weight"],
            ccp_alpha=opts["ccp_alpha"],
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
            best_val = v

    # Store best value
    opts["min_samples_split"] = best_val

    return opts, max_acc


def _optimize_msl(
    X_train_pca, y_train_flat, X_test_pca, y_true, opts
) -> tuple[Any, float]:
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = np.arange(1, 20)
    best_val = variable_array[0]
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = DecisionTreeClassifier(
            random_state=opts["random_state"],
            criterion=opts["criterion"],
            splitter=opts["splitter"],
            max_depth=opts["max_depth"],
            min_samples_split=opts["min_samples_split"],
            min_samples_leaf=v,
            min_weight_fraction_leaf=opts["min_weight_fraction_leaf"],
            max_features=opts["max_features"],
            max_leaf_nodes=opts["max_leaf_nodes"],
            min_impurity_decrease=opts["min_impurity_decrease"],
            class_weight=opts["class_weight"],
            ccp_alpha=opts["ccp_alpha"],
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
            best_val = v

    # Store best value
    opts["min_samples_leaf"] = best_val

    return opts, max_acc


def _optimize_mwfl(
    X_train_pca, y_train_flat, X_test_pca, y_true, opts
) -> tuple[Any, float]:
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = np.arange(0, 0.5, 0.01)
    best_val = variable_array[0]
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = DecisionTreeClassifier(
            random_state=opts["random_state"],
            criterion=opts["criterion"],
            splitter=opts["splitter"],
            max_depth=opts["max_depth"],
            min_samples_split=opts["min_samples_split"],
            min_samples_leaf=opts["min_samples_leaf"],
            min_weight_fraction_leaf=v,
            max_features=opts["max_features"],
            max_leaf_nodes=opts["max_leaf_nodes"],
            min_impurity_decrease=opts["min_impurity_decrease"],
            class_weight=opts["class_weight"],
            ccp_alpha=opts["ccp_alpha"],
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
            best_val = v

    # Store best value
    opts["min_weight_fraction_leaf"] = best_val

    return opts, max_acc


def _optimize_mf(
    X_train_pca, y_train_flat, X_test_pca, y_true, opts
) -> tuple[Any, float]:
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = np.arange(1, 20)
    variable_array = np.append(variable_array.astype(object), ["sqrt", "log2", None])  # type: ignore
    best_val = variable_array[0]
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = DecisionTreeClassifier(
            random_state=opts["random_state"],
            criterion=opts["criterion"],
            splitter=opts["splitter"],
            max_depth=opts["max_depth"],
            min_samples_split=opts["min_samples_split"],
            min_samples_leaf=opts["min_samples_leaf"],
            min_weight_fraction_leaf=opts["min_weight_fraction_leaf"],
            max_features=v,
            max_leaf_nodes=opts["max_leaf_nodes"],
            min_impurity_decrease=opts["min_impurity_decrease"],
            class_weight=opts["class_weight"],
            ccp_alpha=opts["ccp_alpha"],
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
            best_val = v

    # Store best value
    opts["max_features"] = best_val

    return opts, max_acc


def _optimize_mln(
    X_train_pca, y_train_flat, X_test_pca, y_true, opts
) -> tuple[Any, float]:
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = np.arange(2, 50, 1)
    variable_array = np.append(variable_array.astype(object), None)  # type: ignore
    best_val = variable_array[0]
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = DecisionTreeClassifier(
            random_state=opts["random_state"],
            criterion=opts["criterion"],
            splitter=opts["splitter"],
            max_depth=opts["max_depth"],
            min_samples_split=opts["min_samples_split"],
            min_samples_leaf=opts["min_samples_leaf"],
            min_weight_fraction_leaf=opts["min_weight_fraction_leaf"],
            max_features=opts["max_features"],
            max_leaf_nodes=v,
            min_impurity_decrease=opts["min_impurity_decrease"],
            class_weight=opts["class_weight"],
            ccp_alpha=opts["ccp_alpha"],
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
            best_val = v

    # Store best value
    opts["max_leaf_nodes"] = best_val

    return opts, max_acc


def _optimize_mid(
    X_train_pca, y_train_flat, X_test_pca, y_true, opts
) -> tuple[Any, float]:
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = np.arange(0, 1, 0.02)
    best_val = variable_array[0]
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = DecisionTreeClassifier(
            random_state=opts["random_state"],
            criterion=opts["criterion"],
            splitter=opts["splitter"],
            max_depth=opts["max_depth"],
            min_samples_split=opts["min_samples_split"],
            min_samples_leaf=opts["min_samples_leaf"],
            min_weight_fraction_leaf=opts["min_weight_fraction_leaf"],
            max_features=opts["max_features"],
            max_leaf_nodes=opts["max_leaf_nodes"],
            min_impurity_decrease=v,
            class_weight=opts["class_weight"],
            ccp_alpha=opts["ccp_alpha"],
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
            best_val = v

    # Store best value
    opts["min_impurity_decrease"] = best_val

    return opts, max_acc


def _optimize_cw(
    X_train_pca, y_train_flat, X_test_pca, y_true, opts
) -> tuple[Any, float]:
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = ["balanced", None]
    best_val = variable_array[0]
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = DecisionTreeClassifier(
            random_state=opts["random_state"],
            criterion=opts["criterion"],
            splitter=opts["splitter"],
            max_depth=opts["max_depth"],
            min_samples_split=opts["min_samples_split"],
            min_samples_leaf=opts["min_samples_leaf"],
            min_weight_fraction_leaf=opts["min_weight_fraction_leaf"],
            max_features=opts["max_features"],
            max_leaf_nodes=opts["max_leaf_nodes"],
            min_impurity_decrease=opts["min_impurity_decrease"],
            class_weight=v,
            ccp_alpha=opts["ccp_alpha"],
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
            best_val = v

    # Store best value
    opts["class_weight"] = best_val

    return opts, max_acc


def _optimize_decision_tree(X_train_pca, y_train_flat, X_test_pca, y_true, cycles=2):

    # Shorten parameters
    Xtr_pca = X_train_pca
    ytr_flat = y_train_flat
    Xte_pca = X_test_pca

    # Define optimals
    opts = {
        "random_state": 42,
        "criterion": "gini",
        "splitter": "best",
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "min_weight_fraction_leaf": 0.0,
        "max_features": None,
        "max_leaf_nodes": None,
        "min_impurity_decrease": 0.0,
        "class_weight": None,
        "ccp_alpha": 0.0,
        "monotonic_cst": None,
    }

    # Cyclically optimize hyperparameters
    ma_vec = []
    for c in np.arange(cycles):
        opts, _ = _optimize_rs(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _ = _optimize_ct(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _ = _optimize_sp(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _ = _optimize_md(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _ = _optimize_mss(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _ = _optimize_msl(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _ = _optimize_mwfl(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _ = _optimize_mf(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _ = _optimize_mln(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _ = _optimize_mid(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _ = _optimize_cw(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, ma = _optimize_rs(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        ma_vec.append(ma)

    return opts, ma_vec
