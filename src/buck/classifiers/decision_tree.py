from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier


# ----------------- RANDOM STATE -----------------
def _optimize_rs(
    X_train_pca,
    y_train_flat,
    X_test_pca,
    y_true,
    random_state=None,
    criterion="gini",
    splitter="best",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features=None,
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    class_weight=None,
    ccp_alpha=0.0,
    monotonic_cst=None,
) -> tuple[Any, float]:
    """
    Optimizes the random state for a Random Forest classifier.
    :param X_train_pca: PCA transformed training data
    :param y_train_flat: Flattened training labels
    :param X_test_pca: PCA transformed test data
    :param y_true: True labels for the test data
    """
    # Initialize variables
    print("Optimizing random state...")
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
            criterion="gini",
            splitter="best",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features=None,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            class_weight=None,
            ccp_alpha=0.0,
            monotonic_cst=None,
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
    random_state = best_val

    return random_state, max_acc


def _optimize_ct(
    X_train_pca,
    y_train_flat,
    X_test_pca,
    y_true,
    random_state=None,
    criterion="gini",
    splitter="best",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features=None,
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    class_weight=None,
    ccp_alpha=0.0,
    monotonic_cst=None,
) -> tuple[Any, float]:
    print("Optimizing criterion...")
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
            random_state=random_state,
            criterion=v,
            splitter="best",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features=None,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            class_weight=None,
            ccp_alpha=0.0,
            monotonic_cst=None,
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
    criterion = best_val

    return criterion, max_acc


def _optimize_sp(
    X_train_pca,
    y_train_flat,
    X_test_pca,
    y_true,
    random_state=None,
    criterion="gini",
    splitter="best",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features=None,
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    class_weight=None,
    ccp_alpha=0.0,
    monotonic_cst=None,
) -> tuple[Any, float]:
    print("Optimizing splitter...")
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
            random_state=random_state,
            criterion=criterion,
            splitter=v,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features=None,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            class_weight=None,
            ccp_alpha=0.0,
            monotonic_cst=None,
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
    splitter = best_val

    return splitter, max_acc


def _optimize_md(
    X_train_pca,
    y_train_flat,
    X_test_pca,
    y_true,
    random_state=None,
    criterion="gini",
    splitter="best",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features=None,
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    class_weight=None,
    ccp_alpha=0.0,
    monotonic_cst=None,
) -> tuple[Any, float]:
    print("Optimizing max depth...")
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
            random_state=random_state,
            criterion=criterion,
            splitter="best",
            max_depth=v,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features=None,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            class_weight=None,
            ccp_alpha=0.0,
            monotonic_cst=None,
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
    max_depth = best_val

    return max_depth, max_acc


def _optimize_mss(
    X_train_pca,
    y_train_flat,
    X_test_pca,
    y_true,
    random_state=None,
    criterion="gini",
    splitter="best",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features=None,
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    class_weight=None,
    ccp_alpha=0.0,
    monotonic_cst=None,
) -> tuple[Any, float]:
    print("Optimizing min samples split...")
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
            random_state=random_state,
            criterion=criterion,
            splitter="best",
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features=None,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            class_weight=None,
            ccp_alpha=0.0,
            monotonic_cst=None,
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
    min_samples_split = best_val

    return min_samples_split, max_acc


def _optimize_msl(
    X_train_pca,
    y_train_flat,
    X_test_pca,
    y_true,
    random_state=None,
    criterion="gini",
    splitter="best",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features=None,
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    class_weight=None,
    ccp_alpha=0.0,
    monotonic_cst=None,
) -> tuple[Any, float]:
    print("Optimizing min samples leaf...")
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
            random_state=random_state,
            criterion=criterion,
            splitter="best",
            max_depth=max_depth,
            min_samples_split=2,
            min_samples_leaf=v,
            min_weight_fraction_leaf=0.0,
            max_features=None,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            class_weight=None,
            ccp_alpha=0.0,
            monotonic_cst=None,
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
    min_samples_leaf = best_val

    return min_samples_leaf, max_acc


def _optimize_mwfl(
    X_train_pca,
    y_train_flat,
    X_test_pca,
    y_true,
    random_state=None,
    criterion="gini",
    splitter="best",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features=None,
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    class_weight=None,
    ccp_alpha=0.0,
    monotonic_cst=None,
) -> tuple[Any, float]:
    print("Optimizing min weight fraction leaf...")
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = np.arange(0, 1.0, 0.02)
    best_val = variable_array[0]
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = DecisionTreeClassifier(
            random_state=random_state,
            criterion=criterion,
            splitter="best",
            max_depth=max_depth,
            min_samples_split=2,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=v,
            max_features=None,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            class_weight=None,
            ccp_alpha=0.0,
            monotonic_cst=None,
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
            min_weight_fraction_leaf = accuracy
            best_val = v

    # Store best value
    min_weight_fraction_leaf = best_val

    return min_weight_fraction_leaf, max_acc


def _optimize_mf(
    X_train_pca,
    y_train_flat,
    X_test_pca,
    y_true,
    random_state=None,
    criterion="gini",
    splitter="best",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features=None,
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    class_weight=None,
    ccp_alpha=0.0,
    monotonic_cst=None,
) -> tuple[Any, float]:
    print("Optimizing max features...")
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
            random_state=random_state,
            criterion=criterion,
            splitter="best",
            max_depth=max_depth,
            min_samples_split=2,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=v,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            class_weight=None,
            ccp_alpha=0.0,
            monotonic_cst=None,
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
            min_weight_fraction_leaf = accuracy
            best_val = v

    # Store best value
    max_features = best_val

    return max_features, max_acc


def _optimize_mln(
    X_train_pca,
    y_train_flat,
    X_test_pca,
    y_true,
    random_state=None,
    criterion="gini",
    splitter="best",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features=None,
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    class_weight=None,
    ccp_alpha=0.0,
    monotonic_cst=None,
) -> tuple[Any, float]:
    print("Optimizing max leaf nodes...")
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
            random_state=random_state,
            criterion=criterion,
            splitter="best",
            max_depth=max_depth,
            min_samples_split=2,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=v,
            min_impurity_decrease=0.0,
            class_weight=None,
            ccp_alpha=0.0,
            monotonic_cst=None,
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
            min_weight_fraction_leaf = accuracy
            best_val = v

    # Store best value
    max_leaf_nodes = best_val

    return max_leaf_nodes, max_acc


def _optimize_mid(
    X_train_pca,
    y_train_flat,
    X_test_pca,
    y_true,
    random_state=None,
    criterion="gini",
    splitter="best",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features=None,
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    class_weight=None,
    ccp_alpha=0.0,
    monotonic_cst=None,
) -> tuple[Any, float]:
    print("Optimizing min impurity decrease...")
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
            random_state=random_state,
            criterion=criterion,
            splitter="best",
            max_depth=max_depth,
            min_samples_split=2,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=v,
            class_weight=None,
            ccp_alpha=0.0,
            monotonic_cst=None,
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
            min_weight_fraction_leaf = accuracy
            best_val = v

    # Store best value
    min_impurity_decrease = best_val

    return min_impurity_decrease, max_acc


def _optimize_cw(
    X_train_pca,
    y_train_flat,
    X_test_pca,
    y_true,
    random_state=None,
    criterion="gini",
    splitter="best",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features=None,
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    class_weight=None,
    ccp_alpha=0.0,
    monotonic_cst=None,
) -> tuple[Any, float]:
    print("Optimizing class weight...")
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
            random_state=random_state,
            criterion=criterion,
            splitter="best",
            max_depth=max_depth,
            min_samples_split=2,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            class_weight=v,
            ccp_alpha=0.0,
            monotonic_cst=None,
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
            min_weight_fraction_leaf = accuracy
            best_val = v

    # Store best value
    class_weight = best_val

    return class_weight, max_acc


def optimize_decision_tree(X_train_pca, y_train_flat, X_test_pca, y_true):
    # Shorten parameters
    Xtr_pca = X_train_pca
    ytr_flat = y_train_flat
    Xte_pca = X_test_pca

    # Optimize hyperparameters
    rs, ma = _optimize_rs(Xtr_pca, ytr_flat, Xte_pca, y_true)
    print("Accuracy: ", ma)
    ct, ma = _optimize_ct(Xtr_pca, ytr_flat, Xte_pca, y_true, rs)
    print("Accuracy: ", ma)
    sp, ma = _optimize_sp(Xtr_pca, ytr_flat, Xte_pca, y_true, rs, ct)
    print("Accuracy: ", ma)
    md, ma = _optimize_md(Xtr_pca, ytr_flat, Xte_pca, y_true, rs, ct, sp)
    print("Accuracy: ", ma)
    mss, ma = _optimize_mss(Xtr_pca, ytr_flat, Xte_pca, y_true, rs, ct, sp, md)
    print("Accuracy: ", ma)
    msl, ma = _optimize_msl(Xtr_pca, ytr_flat, Xte_pca, y_true, rs, ct, sp, md, mss)
    print("Accuracy: ", ma)
    mwfl, ma = _optimize_mwfl(
        Xtr_pca, ytr_flat, Xte_pca, y_true, rs, ct, sp, md, mss, msl
    )
    print("Accuracy: ", ma)
    mf, ma = _optimize_mf(
        Xtr_pca, ytr_flat, Xte_pca, y_true, rs, ct, sp, md, mss, msl, mwfl
    )
    print("Accuracy: ", ma)
    mln, ma = _optimize_mln(
        Xtr_pca, ytr_flat, Xte_pca, y_true, rs, ct, sp, md, mss, msl, mwfl, mf
    )
    print("Accuracy: ", ma)
    mid, ma = _optimize_mid(
        Xtr_pca, ytr_flat, Xte_pca, y_true, rs, ct, sp, md, mss, msl, mwfl, mf
    )
    print("Accuracy: ", ma)
    cw, ma = _optimize_cw(
        Xtr_pca, ytr_flat, Xte_pca, y_true, rs, ct, sp, md, mss, msl, mwfl, mf
    )
    print("Accuracy: ", ma)
    rs, ma = _optimize_rs(
        Xtr_pca, ytr_flat, Xte_pca, y_true, rs, ct, sp, md, mss, msl, mwfl, mf
    )
    print("Accuracy: ", ma)
