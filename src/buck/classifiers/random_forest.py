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
    random_state=42,
    n_estimators=140,
    max_depth=5,
    criterion="gini",
    class_weight="balanced",
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features="sqrt",
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    bootstrap=True,
    oob_score=False,
    n_jobs=-1,
    verbose=0,
    warm_start=False,
    ccp_alpha=0.0,
    max_samples=None,
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
    for v in variable_array:
        # Define classifiers to test
        classifier = RandomForestClassifier(
            random_state=v,
            n_estimators=140,
            max_depth=5,
            criterion="gini",
            class_weight="balanced",
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features="sqrt",
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            bootstrap=True,
            oob_score=False,
            n_jobs=-1,
            verbose=0,
            warm_start=False,
            ccp_alpha=0.0,
            max_samples=None,
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


def _optimize_nest(
    X_train_pca,
    y_train_flat,
    X_test_pca,
    y_true,
    random_state=42,
    n_estimators=140,
    max_depth=5,
    criterion="gini",
    class_weight="balanced",
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features="sqrt",
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    bootstrap=True,
    oob_score=False,
    n_jobs=-1,
    verbose=0,
    warm_start=False,
    ccp_alpha=0.0,
    max_samples=None,
    monotonic_cst=None,
) -> tuple[Any, float]:
    """
    Optimizes the number of estimators for a Random Forest classifier.
    :param X_train_pca: PCA transformed training data
    :param y_train_flat: Flattened training labels
    :param X_test_pca: PCA transformed test data
    :param y_true: True labels for the test data
    :param random_state: Random state for reproducibility
    """
    print("Optimizing # estimators...")

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
        classifier = RandomForestClassifier(
            random_state=random_state,
            n_estimators=v,
            max_depth=5,
            criterion="gini",
            class_weight="balanced",
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features="sqrt",
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            bootstrap=True,
            oob_score=False,
            n_jobs=-1,
            verbose=0,
            warm_start=False,
            ccp_alpha=0.0,
            max_samples=None,
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
    n_estimators = best_val

    return n_estimators, max_acc


def _optimize_max_d(
    X_train_pca,
    y_train_flat,
    X_test_pca,
    y_true,
    random_state=42,
    n_estimators=140,
    max_depth=5,
    criterion="gini",
    class_weight="balanced",
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features="sqrt",
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    bootstrap=True,
    oob_score=False,
    n_jobs=-1,
    verbose=0,
    warm_start=False,
    ccp_alpha=0.0,
    max_samples=None,
    monotonic_cst=None,
) -> tuple[Any, float]:
    """
    Optimizes the maximum depth for a Random Forest classifier.
    :param X_train_pca: PCA transformed training data
    :param y_train_flat: Flattened training labels
    :param X_test_pca: PCA transformed test data
    :param y_true: True labels for the test data
    :param random_state: Random state for reproducibility
    """
    print("Optimizing max depth...")

    # Initialize variables
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = np.arange(1, 15)
    variable_array = np.append(variable_array.astype(object), None)  # type: ignore
    best_val = variable_array[0]

    # Iterate through variables
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = RandomForestClassifier(
            random_state=random_state,
            n_estimators=n_estimators,
            max_depth=v,
            criterion="gini",
            class_weight="balanced",
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features="sqrt",
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            bootstrap=True,
            oob_score=False,
            n_jobs=-1,
            verbose=0,
            warm_start=False,
            ccp_alpha=0.0,
            max_samples=None,
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


def _optimize_crit(
    X_train_pca,
    y_train_flat,
    X_test_pca,
    y_true,
    random_state,
    n_estimators,
    max_depth,
    criterion="gini",
    class_weight="balanced",
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features="sqrt",
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    bootstrap=True,
    oob_score=False,
    n_jobs=-1,
    verbose=0,
    warm_start=False,
    ccp_alpha=0.0,
    max_samples=None,
    monotonic_cst=None,
) -> tuple[str, float]:
    """
    Optimizes the criterion for a Random Forest classifier.
    :param X_train_pca: PCA transformed training data
    :param y_train_flat: Flattened training labels
    :param X_test_pca: PCA transformed test data
    :param y_true: True labels for the test data
    :param random_state: Random state for reproducibility
    """
    print("Optimizing criterion...")

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
            random_state=random_state,
            n_estimators=n_estimators,
            max_depth=max_depth,
            criterion=v,
            class_weight="balanced",
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features="sqrt",
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            bootstrap=True,
            oob_score=False,
            n_jobs=-1,
            verbose=0,
            warm_start=False,
            ccp_alpha=0.0,
            max_samples=None,
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


def _optimize_cw(
    X_train_pca,
    y_train_flat,
    X_test_pca,
    y_true,
    random_state,
    n_estimators,
    max_depth,
    criterion,
    class_weight="balanced",
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features="sqrt",
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    bootstrap=True,
    oob_score=False,
    n_jobs=-1,
    verbose=0,
    warm_start=False,
    ccp_alpha=0.0,
    max_samples=None,
    monotonic_cst=None,
):
    """
    Optimizes the class weight for a Random Forest classifier.
    :param X_train_pca: PCA transformed training data
    :param y_train_flat: Flattened training labels
    :param X_test_pca: PCA transformed test data
    :param y_true: True labels for the test data
    :param random_state: Random state for reproducibility
    """
    print("Optimizing class weight...")

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
            random_state=random_state,
            n_estimators=n_estimators,
            max_depth=max_depth,
            criterion=criterion,
            class_weight=v,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features="sqrt",
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            bootstrap=True,
            oob_score=False,
            n_jobs=-1,
            verbose=0,
            warm_start=False,
            ccp_alpha=0.0,
            max_samples=None,
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
    class_weight = best_val

    return class_weight, max_acc


def _optimize_mss(
    X_train_pca,
    y_train_flat,
    X_test_pca,
    y_true,
    random_state,
    n_estimators,
    max_depth,
    criterion,
    class_weight,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features="sqrt",
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    bootstrap=True,
    oob_score=False,
    n_jobs=-1,
    verbose=0,
    warm_start=False,
    ccp_alpha=0.0,
    max_samples=None,
    monotonic_cst=None,
) -> tuple[Any, float]:
    """
    Optimizes the minimum samples split for a Random Forest classifier.
    :param X_train_pca: PCA transformed training data
    :param y_train_flat: Flattened training labels
    :param X_test_pca: PCA transformed test data
    :param y_true: True labels for the test data
    :param random_state: Random state for reproducibility
    """
    print("Optimizing min samples split...")

    # Initialize variables
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = np.arange(2, 30)
    best_val = variable_array[0]

    # Iterate through variables
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = RandomForestClassifier(
            random_state=random_state,
            n_estimators=n_estimators,
            max_depth=max_depth,
            criterion=criterion,
            class_weight=class_weight,
            min_samples_split=v,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features="sqrt",
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            bootstrap=True,
            oob_score=False,
            n_jobs=-1,
            verbose=0,
            warm_start=False,
            ccp_alpha=0.0,
            max_samples=None,
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
    random_state,
    n_estimators,
    max_depth,
    criterion,
    class_weight,
    min_samples_split,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features="sqrt",
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    bootstrap=True,
    oob_score=False,
    n_jobs=-1,
    verbose=0,
    warm_start=False,
    ccp_alpha=0.0,
    max_samples=None,
    monotonic_cst=None,
) -> tuple[Any, float]:
    """
    Optimizes the minimum samples leaf for a Random Forest classifier.
    :param X_train_pca: PCA transformed training data
    :param y_train_flat: Flattened training labels
    :param X_test_pca: PCA transformed test data
    :param y_true: True labels for the test data
    :param random_state: Random state for reproducibility
    """
    print("Optimizing min samples leaf...")

    # Initialize variables
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = np.arange(1, 20)
    best_val = variable_array[0]

    # Iterate through variables
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = RandomForestClassifier(
            random_state=random_state,
            n_estimators=n_estimators,
            max_depth=max_depth,
            criterion=criterion,
            class_weight=class_weight,
            min_samples_split=min_samples_split,
            min_samples_leaf=v,
            min_weight_fraction_leaf=0.0,
            max_features="sqrt",
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            bootstrap=True,
            oob_score=False,
            n_jobs=-1,
            verbose=0,
            warm_start=False,
            ccp_alpha=0.0,
            max_samples=None,
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
    random_state,
    n_estimators,
    max_depth,
    criterion,
    class_weight,
    min_samples_split,
    min_samples_leaf,
    min_weight_fraction_leaf=0.0,
    max_features="sqrt",
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    bootstrap=True,
    oob_score=False,
    n_jobs=-1,
    verbose=0,
    warm_start=False,
    ccp_alpha=0.0,
    max_samples=None,
    monotonic_cst=None,
) -> tuple[Any, float]:
    """
    Optimizes the minimum weight fraction leaf for a Random Forest classifier.
    :param X_train_pca: PCA transformed training data
    :param y_train_flat: Flattened training labels
    :param X_test_pca: PCA transformed test data
    :param y_true: True labels for the test data
    :param random_state: Random state for reproducibility
    """
    print("Optimizing min weight fraction leaf...")

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
            random_state=random_state,
            n_estimators=n_estimators,
            max_depth=max_depth,
            criterion=criterion,
            class_weight=class_weight,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=v,
            max_features="sqrt",
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            bootstrap=True,
            oob_score=False,
            n_jobs=-1,
            verbose=0,
            warm_start=False,
            ccp_alpha=0.0,
            max_samples=None,
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
    min_weight_fraction_leaf = best_val

    return min_weight_fraction_leaf, max_acc


def _optimize_mfeat(
    X_train_pca,
    y_train_flat,
    X_test_pca,
    y_true,
    random_state,
    n_estimators,
    max_depth,
    criterion,
    class_weight,
    min_samples_split,
    min_samples_leaf,
    min_weight_fraction_leaf,
    max_features="sqrt",
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    bootstrap=True,
    oob_score=False,
    n_jobs=-1,
    verbose=0,
    warm_start=False,
    ccp_alpha=0.0,
    max_samples=None,
    monotonic_cst=None,
):
    """
    Optimizes the maximum features for a Random Forest classifier.
    :param X_train_pca: PCA transformed training data
    :param y_train_flat: Flattened training labels
    :param X_test_pca: PCA transformed test data
    :param y_true: True labels for the test data
    :param random_state: Random state for reproducibility
    """
    print("Optimizing max features...")

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
            random_state=random_state,
            n_estimators=n_estimators,
            max_depth=max_depth,
            criterion=criterion,
            class_weight=class_weight,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=v,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            bootstrap=True,
            oob_score=False,
            n_jobs=-1,
            verbose=0,
            warm_start=False,
            ccp_alpha=0.0,
            max_samples=None,
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
    max_features = best_val

    return max_features, max_acc


def _optimize_mln(
    X_train_pca,
    y_train_flat,
    X_test_pca,
    y_true,
    random_state,
    n_estimators,
    max_depth,
    criterion,
    class_weight,
    min_samples_split,
    min_samples_leaf,
    min_weight_fraction_leaf,
    max_features,
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    bootstrap=True,
    oob_score=False,
    n_jobs=-1,
    verbose=0,
    warm_start=False,
    ccp_alpha=0.0,
    max_samples=None,
    monotonic_cst=None,
) -> tuple[Any, float]:
    """
    Optimizes the maximum leaf nodes for a Random Forest classifier.
    :param X_train_pca: PCA transformed training data
    :param y_train_flat: Flattened training labels
    :param X_test_pca: PCA transformed test data
    :param y_true: True labels for the test data
    :param random_state: Random state for reproducibility
    """
    print("Optimizing max leaf nodes...")

    # Initialize variables
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = np.arange(2, 20)
    variable_array = np.append(variable_array.astype(object), None)  # type: ignore
    best_val = variable_array[0]

    # Iterate through variables
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = RandomForestClassifier(
            random_state=random_state,
            n_estimators=n_estimators,
            max_depth=max_depth,
            criterion=criterion,
            class_weight=class_weight,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=v,
            min_impurity_decrease=0.0,
            bootstrap=True,
            oob_score=False,
            n_jobs=-1,
            verbose=0,
            warm_start=False,
            ccp_alpha=0.0,
            max_samples=None,
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
    max_leaf_nodes = best_val

    return max_leaf_nodes, max_acc


def _optimize_mid(
    X_train_pca,
    y_train_flat,
    X_test_pca,
    y_true,
    random_state,
    n_estimators,
    max_depth,
    criterion,
    class_weight,
    min_samples_split,
    min_samples_leaf,
    min_weight_fraction_leaf,
    max_features,
    max_leaf_nodes,
    min_impurity_decrease=0.0,
    bootstrap=True,
    oob_score=False,
    n_jobs=-1,
    verbose=0,
    warm_start=False,
    ccp_alpha=0.0,
    max_samples=None,
    monotonic_cst=None,
) -> tuple[Any, float]:
    """
    Optimizes the minimum impurity decrease for a Random Forest classifier.
    :param X_train_pca: PCA transformed training data
    :param y_train_flat: Flattened training labels
    :param X_test_pca: PCA transformed test data
    :param y_true: True labels for the test data
    :param random_state: Random state for reproducibility
    """
    print("Optimizing min impurity decrease...")

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
            random_state=random_state,
            n_estimators=n_estimators,
            max_depth=max_depth,
            criterion=criterion,
            class_weight=class_weight,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=v,
            bootstrap=True,
            oob_score=False,
            n_jobs=-1,
            verbose=0,
            warm_start=False,
            ccp_alpha=0.0,
            max_samples=None,
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
    min_impurity_decrease = best_val

    return min_impurity_decrease, max_acc


def optimize_random_forest(
    X_train_pca,
    y_train_flat,
    X_test_pca,
    y_true,
) -> None:
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

    # Optimize hyperparameters
    rs, ma = _optimize_rs(Xtr_pca, ytr_flat, Xte_pca, y_true)
    print("Accuracy: ", ma)
    n_est, ma = _optimize_nest(Xtr_pca, ytr_flat, Xte_pca, y_true, random_state=rs)
    print("Accuracy: ", ma)
    max_d, ma = _optimize_max_d(
        Xtr_pca, ytr_flat, Xte_pca, y_true, random_state=rs, n_estimators=n_est
    )
    print("Accuracy: ", ma)
    crit, ma = _optimize_crit(
        Xtr_pca,
        ytr_flat,
        Xte_pca,
        y_true,
        random_state=rs,
        n_estimators=n_est,
        max_depth=max_d,
    )
    print("Accuracy: ", ma)
    c_wt, ma = _optimize_cw(
        Xtr_pca,
        ytr_flat,
        Xte_pca,
        y_true,
        random_state=rs,
        n_estimators=n_est,
        max_depth=max_d,
        criterion=crit,
    )
    print("Accuracy: ", ma)
    m_spl, ma = _optimize_mss(
        Xtr_pca,
        ytr_flat,
        Xte_pca,
        y_true,
        random_state=rs,
        n_estimators=n_est,
        max_depth=max_d,
        criterion=crit,
        class_weight=c_wt,
    )
    print("Accuracy: ", ma)
    m_sl, ma = _optimize_msl(
        Xtr_pca,
        ytr_flat,
        Xte_pca,
        y_true,
        random_state=rs,
        n_estimators=n_est,
        max_depth=max_d,
        criterion=crit,
        class_weight=c_wt,
        min_samples_split=m_spl,
    )
    print("Accuracy: ", ma)
    m_wfl, ma = _optimize_mwfl(
        Xtr_pca,
        ytr_flat,
        Xte_pca,
        y_true,
        random_state=rs,
        n_estimators=n_est,
        max_depth=max_d,
        criterion=crit,
        class_weight=c_wt,
        min_samples_split=m_spl,
        min_samples_leaf=m_sl,
    )
    print("Accuracy: ", ma)
    m_feat, ma = _optimize_mfeat(
        Xtr_pca,
        ytr_flat,
        Xte_pca,
        y_true,
        random_state=rs,
        n_estimators=n_est,
        max_depth=max_d,
        criterion=crit,
        class_weight=c_wt,
        min_samples_split=m_spl,
        min_samples_leaf=m_sl,
        min_weight_fraction_leaf=m_wfl,
    )
    print("Accuracy: ", ma)
    m_ln, ma = _optimize_mln(
        Xtr_pca,
        ytr_flat,
        Xte_pca,
        y_true,
        random_state=rs,
        n_estimators=n_est,
        max_depth=max_d,
        criterion=crit,
        class_weight=c_wt,
        min_samples_split=m_spl,
        min_samples_leaf=m_sl,
        min_weight_fraction_leaf=m_wfl,
        max_features=m_feat,
    )
    print("Accuracy: ", ma)
    m_id, ma = _optimize_mid(
        Xtr_pca,
        ytr_flat,
        Xte_pca,
        y_true,
        random_state=rs,
        n_estimators=n_est,
        max_depth=max_d,
        criterion=crit,
        class_weight=c_wt,
        min_samples_split=m_spl,
        min_samples_leaf=m_sl,
        min_weight_fraction_leaf=m_wfl,
        max_features=m_feat,
        max_leaf_nodes=m_ln,
    )
    print("Accuracy: ", ma)

    rs, ma = _optimize_rs(
        Xtr_pca,
        ytr_flat,
        Xte_pca,
        y_true,
        n_estimators=n_est,
        max_depth=max_d,
        criterion=crit,
        class_weight=c_wt,
        min_samples_split=m_spl,
        min_samples_leaf=m_sl,
        min_weight_fraction_leaf=m_wfl,
        max_features=m_feat,
        max_leaf_nodes=m_ln,
        min_impurity_decrease=m_id,
    )
    print("Accuracy: ", ma)

    n_est, ma = _optimize_nest(
        Xtr_pca,
        ytr_flat,
        Xte_pca,
        y_true,
        random_state=rs,
        max_depth=max_d,
        criterion=crit,
        class_weight=c_wt,
        min_samples_split=m_spl,
        min_samples_leaf=m_sl,
        min_weight_fraction_leaf=m_wfl,
        max_features=m_feat,
        max_leaf_nodes=m_ln,
        min_impurity_decrease=m_id,
    )
    print("Accuracy: ", ma)
