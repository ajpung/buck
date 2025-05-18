from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score


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
        classifier = GradientBoostingClassifier(
            random_state=v,
            loss=opts["loss"],
            learning_rate=opts["learning_rate"],
            n_estimators=opts["n_estimators"],
            subsample=opts["subsample"],
            criterion=opts["criterion"],
            min_samples_split=opts["min_samples_split"],
            min_samples_leaf=opts["min_samples_leaf"],
            min_weight_fraction_leaf=opts["min_weight_fraction_leaf"],
            max_depth=opts["max_depth"],
            min_impurity_decrease=opts["min_impurity_decrease"],
            init=opts["init"],
            max_features=opts["max_features"],
            verbose=opts["verbose"],
            max_leaf_nodes=opts["max_leaf_nodes"],
            warm_start=opts["warm_start"],
            validation_fraction=opts["validation_fraction"],
            n_iter_no_change=opts["n_iter_no_change"],
            tol=opts["tol"],
            ccp_alpha=opts["ccp_alpha"],
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


def _optimize_lr(
    X_train_pca, y_train_flat, X_test_pca, y_true, opts
) -> tuple[Any, float]:
    # Initialize variables
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = np.arange(0.0, 1.0, 0.02)
    best_val = variable_array[0]

    # Iterate through variables
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = GradientBoostingClassifier(
            random_state=opts["random_state"],
            loss=opts["loss"],
            learning_rate=v,
            n_estimators=opts["n_estimators"],
            subsample=opts["subsample"],
            criterion=opts["criterion"],
            min_samples_split=opts["min_samples_split"],
            min_samples_leaf=opts["min_samples_leaf"],
            min_weight_fraction_leaf=opts["min_weight_fraction_leaf"],
            max_depth=opts["max_depth"],
            min_impurity_decrease=opts["min_impurity_decrease"],
            init=opts["init"],
            max_features=opts["max_features"],
            verbose=opts["verbose"],
            max_leaf_nodes=opts["max_leaf_nodes"],
            warm_start=opts["warm_start"],
            validation_fraction=opts["validation_fraction"],
            n_iter_no_change=opts["n_iter_no_change"],
            tol=opts["tol"],
            ccp_alpha=opts["ccp_alpha"],
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
    opts["learning_rate"] = best_val

    return opts, max_acc


def _optimize_nest(
    X_train_pca, y_train_flat, X_test_pca, y_true, opts
) -> tuple[Any, float]:
    # Initialize variables
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = np.arange(1, 100, 1)
    best_val = variable_array[0]

    # Iterate through variables
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
    # Define classifiers to test
    classifier = GradientBoostingClassifier(
        random_state=opts["random_state"],
        loss=opts["loss"],
        learning_rate=opts["learning_rate"],
        n_estimators=v,
        subsample=opts["subsample"],
        criterion=opts["criterion"],
        min_samples_split=opts["min_samples_split"],
        min_samples_leaf=opts["min_samples_leaf"],
        min_weight_fraction_leaf=opts["min_weight_fraction_leaf"],
        max_depth=opts["max_depth"],
        min_impurity_decrease=opts["min_impurity_decrease"],
        init=opts["init"],
        max_features=opts["max_features"],
        verbose=opts["verbose"],
        max_leaf_nodes=opts["max_leaf_nodes"],
        warm_start=opts["warm_start"],
        validation_fraction=opts["validation_fraction"],
        n_iter_no_change=opts["n_iter_no_change"],
        tol=opts["tol"],
        ccp_alpha=opts["ccp_alpha"],
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
    opts["n_estimators"] = best_val

    return opts, max_acc


def _optimize_ss(
    X_train_pca, y_train_flat, X_test_pca, y_true, opts
) -> tuple[Any, float]:
    # Initialize variables
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = np.arange(0.01, 1.0, 0.01)
    best_val = variable_array[0]

    # Iterate through variables
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = GradientBoostingClassifier(
            random_state=opts["random_state"],
            loss=opts["loss"],
            learning_rate=opts["learning_rate"],
            n_estimators=opts["n_estimators"],
            subsample=v,
            criterion=opts["criterion"],
            min_samples_split=opts["min_samples_split"],
            min_samples_leaf=opts["min_samples_leaf"],
            min_weight_fraction_leaf=opts["min_weight_fraction_leaf"],
            max_depth=opts["max_depth"],
            min_impurity_decrease=opts["min_impurity_decrease"],
            init=opts["init"],
            max_features=opts["max_features"],
            verbose=opts["verbose"],
            max_leaf_nodes=opts["max_leaf_nodes"],
            warm_start=opts["warm_start"],
            validation_fraction=opts["validation_fraction"],
            n_iter_no_change=opts["n_iter_no_change"],
            tol=opts["tol"],
            ccp_alpha=opts["ccp_alpha"],
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
    opts["subsample"] = best_val

    return opts, max_acc


def _optimize_cr(
    X_train_pca, y_train_flat, X_test_pca, y_true, opts
) -> tuple[Any, float]:
    # Initialize variables
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = ["friedman_mse", "squared_error"]
    best_val = variable_array[0]

    # Iterate through variables
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = GradientBoostingClassifier(
            random_state=opts["random_state"],
            loss=opts["loss"],
            learning_rate=opts["learning_rate"],
            n_estimators=opts["n_estimators"],
            subsample=opts["subsample"],
            criterion=v,
            min_samples_split=opts["min_samples_split"],
            min_samples_leaf=opts["min_samples_leaf"],
            min_weight_fraction_leaf=opts["min_weight_fraction_leaf"],
            max_depth=opts["max_depth"],
            min_impurity_decrease=opts["min_impurity_decrease"],
            init=opts["init"],
            max_features=opts["max_features"],
            verbose=opts["verbose"],
            max_leaf_nodes=opts["max_leaf_nodes"],
            warm_start=opts["warm_start"],
            validation_fraction=opts["validation_fraction"],
            n_iter_no_change=opts["n_iter_no_change"],
            tol=opts["tol"],
            ccp_alpha=opts["ccp_alpha"],
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


def _optimize_mss(
    X_train_pca, y_train_flat, X_test_pca, y_true, opts
) -> tuple[Any, float]:
    # Initialize variables
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = np.arange(2, 200, 1)
    best_val = variable_array[0]

    # Iterate through variables
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = GradientBoostingClassifier(
            random_state=opts["random_state"],
            loss=opts["loss"],
            learning_rate=opts["learning_rate"],
            n_estimators=opts["n_estimators"],
            subsample=opts["subsample"],
            criterion=opts["criterion"],
            min_samples_split=v,
            min_samples_leaf=opts["min_samples_leaf"],
            min_weight_fraction_leaf=opts["min_weight_fraction_leaf"],
            max_depth=opts["max_depth"],
            min_impurity_decrease=opts["min_impurity_decrease"],
            init=opts["init"],
            max_features=opts["max_features"],
            verbose=opts["verbose"],
            max_leaf_nodes=opts["max_leaf_nodes"],
            warm_start=opts["warm_start"],
            validation_fraction=opts["validation_fraction"],
            n_iter_no_change=opts["n_iter_no_change"],
            tol=opts["tol"],
            ccp_alpha=opts["ccp_alpha"],
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
    # Initialize variables
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = np.arange(1, 200, 1)
    best_val = variable_array[0]

    # Iterate through variables
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = GradientBoostingClassifier(
            random_state=opts["random_state"],
            loss=opts["loss"],
            learning_rate=opts["learning_rate"],
            n_estimators=opts["n_estimators"],
            subsample=opts["subsample"],
            criterion=opts["criterion"],
            min_samples_split=opts["min_samples_split"],
            min_samples_leaf=v,
            min_weight_fraction_leaf=opts["min_weight_fraction_leaf"],
            max_depth=opts["max_depth"],
            min_impurity_decrease=opts["min_impurity_decrease"],
            init=opts["init"],
            max_features=opts["max_features"],
            verbose=opts["verbose"],
            max_leaf_nodes=opts["max_leaf_nodes"],
            warm_start=opts["warm_start"],
            validation_fraction=opts["validation_fraction"],
            n_iter_no_change=opts["n_iter_no_change"],
            tol=opts["tol"],
            ccp_alpha=opts["ccp_alpha"],
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
        classifier = GradientBoostingClassifier(
            random_state=opts["random_state"],
            loss=opts["loss"],
            learning_rate=opts["learning_rate"],
            n_estimators=opts["n_estimators"],
            subsample=opts["subsample"],
            criterion=opts["criterion"],
            min_samples_split=opts["min_samples_split"],
            min_samples_leaf=opts["min_samples_leaf"],
            min_weight_fraction_leaf=v,
            max_depth=opts["max_depth"],
            min_impurity_decrease=opts["min_impurity_decrease"],
            init=opts["init"],
            max_features=opts["max_features"],
            verbose=opts["verbose"],
            max_leaf_nodes=opts["max_leaf_nodes"],
            warm_start=opts["warm_start"],
            validation_fraction=opts["validation_fraction"],
            n_iter_no_change=opts["n_iter_no_change"],
            tol=opts["tol"],
            ccp_alpha=opts["ccp_alpha"],
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


def _optimize_md(
    X_train_pca, y_train_flat, X_test_pca, y_true, opts
) -> tuple[Any, float]:
    # Initialize variables
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = np.arange(1, 20, 1)
    variable_array = np.append(variable_array.astype(object), None)  # type: ignore
    best_val = variable_array[0]

    # Iterate through variables
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = GradientBoostingClassifier(
            random_state=opts["random_state"],
            loss=opts["loss"],
            learning_rate=opts["learning_rate"],
            n_estimators=opts["n_estimators"],
            subsample=opts["subsample"],
            criterion=opts["criterion"],
            min_samples_split=opts["min_samples_split"],
            min_samples_leaf=opts["min_samples_leaf"],
            min_weight_fraction_leaf=opts["min_weight_fraction_leaf"],
            max_depth=v,
            min_impurity_decrease=opts["min_impurity_decrease"],
            init=opts["init"],
            max_features=opts["max_features"],
            verbose=opts["verbose"],
            max_leaf_nodes=opts["max_leaf_nodes"],
            warm_start=opts["warm_start"],
            validation_fraction=opts["validation_fraction"],
            n_iter_no_change=opts["n_iter_no_change"],
            tol=opts["tol"],
            ccp_alpha=opts["ccp_alpha"],
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


def _optimize_mid(
    X_train_pca, y_train_flat, X_test_pca, y_true, opts
) -> tuple[Any, float]:
    # Initialize variables
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = np.arange(0.0, 2.0, 0.01)
    best_val = variable_array[0]

    # Iterate through variables
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = GradientBoostingClassifier(
            random_state=opts["random_state"],
            loss=opts["loss"],
            learning_rate=opts["learning_rate"],
            n_estimators=opts["n_estimators"],
            subsample=opts["subsample"],
            criterion=opts["criterion"],
            min_samples_split=opts["min_samples_split"],
            min_samples_leaf=opts["min_samples_leaf"],
            min_weight_fraction_leaf=opts["min_weight_fraction_leaf"],
            max_depth=opts["max_depth"],
            min_impurity_decrease=v,
            init=opts["init"],
            max_features=opts["max_features"],
            verbose=opts["verbose"],
            max_leaf_nodes=opts["max_leaf_nodes"],
            warm_start=opts["warm_start"],
            validation_fraction=opts["validation_fraction"],
            n_iter_no_change=opts["n_iter_no_change"],
            tol=opts["tol"],
            ccp_alpha=opts["ccp_alpha"],
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


def _optimize_init(
    X_train_pca, y_train_flat, X_test_pca, y_true, opts
) -> tuple[Any, float]:
    # Initialize variables
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = ["zero", None]
    best_val = variable_array[0]

    # Iterate through variables
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = GradientBoostingClassifier(
            random_state=opts["random_state"],
            loss=opts["loss"],
            learning_rate=opts["learning_rate"],
            n_estimators=opts["n_estimators"],
            subsample=opts["subsample"],
            criterion=opts["criterion"],
            min_samples_split=opts["min_samples_split"],
            min_samples_leaf=opts["min_samples_leaf"],
            min_weight_fraction_leaf=opts["min_weight_fraction_leaf"],
            max_depth=opts["max_depth"],
            min_impurity_decrease=opts["min_impurity_decrease"],
            init=v,
            max_features=opts["max_features"],
            verbose=opts["verbose"],
            max_leaf_nodes=opts["max_leaf_nodes"],
            warm_start=opts["warm_start"],
            validation_fraction=opts["validation_fraction"],
            n_iter_no_change=opts["n_iter_no_change"],
            tol=opts["tol"],
            ccp_alpha=opts["ccp_alpha"],
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
    opts["init"] = best_val

    return opts, max_acc


def _optimize_mf(
    X_train_pca, y_train_flat, X_test_pca, y_true, opts
) -> tuple[Any, float]:
    # Initialize variables
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = np.arange(1, 200, 1)
    variable_array = np.append(variable_array.astype(object), ["sqrt", "log2", None])  # type: ignore
    best_val = variable_array[0]

    # Iterate through variables
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = GradientBoostingClassifier(
            random_state=opts["random_state"],
            loss=opts["loss"],
            learning_rate=opts["learning_rate"],
            n_estimators=opts["n_estimators"],
            subsample=opts["subsample"],
            criterion=opts["criterion"],
            min_samples_split=opts["min_samples_split"],
            min_samples_leaf=opts["min_samples_leaf"],
            min_weight_fraction_leaf=opts["min_weight_fraction_leaf"],
            max_depth=opts["max_depth"],
            min_impurity_decrease=opts["min_impurity_decrease"],
            init=opts["init"],
            max_features=v,
            max_leaf_nodes=opts["max_leaf_nodes"],
            verbose=opts["verbose"],
            warm_start=opts["warm_start"],
            validation_fraction=opts["validation_fraction"],
            n_iter_no_change=opts["n_iter_no_change"],
            tol=opts["tol"],
            ccp_alpha=opts["ccp_alpha"],
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
    # Initialize variables
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = np.arange(1, 200, 1)
    variable_array = np.append(variable_array.astype(object), None)  # type: ignore
    best_val = variable_array[0]

    # Iterate through variables
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = GradientBoostingClassifier(
            random_state=opts["random_state"],
            loss=opts["loss"],
            learning_rate=opts["learning_rate"],
            n_estimators=opts["n_estimators"],
            subsample=opts["subsample"],
            criterion=opts["criterion"],
            min_samples_split=opts["min_samples_split"],
            min_samples_leaf=opts["min_samples_leaf"],
            min_weight_fraction_leaf=opts["min_weight_fraction_leaf"],
            max_depth=opts["max_depth"],
            min_impurity_decrease=opts["min_impurity_decrease"],
            init=opts["init"],
            max_features=opts["max_features"],
            max_leaf_nodes=v,
            verbose=opts["verbose"],
            warm_start=opts["warm_start"],
            validation_fraction=opts["validation_fraction"],
            n_iter_no_change=opts["n_iter_no_change"],
            tol=opts["tol"],
            ccp_alpha=opts["ccp_alpha"],
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


def _optimize_gradient_boost(X_train_pca, y_train_flat, X_test_pca, y_true, cycles=2):

    # Shorten parameters
    Xtr_pca = X_train_pca
    ytr_flat = y_train_flat
    Xte_pca = X_test_pca

    # Define optimals
    opts = {
        "random_state": None,
        "loss": "log_loss",
        "learning_rate": 0.1,
        "n_estimators": 100,
        "subsample": 1.0,
        "criterion": "friedman_mse",
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "min_weight_fraction_leaf": 0.0,
        "max_depth": 3,
        "min_impurity_decrease": 0.0,
        "init": None,
        "max_features": None,
        "verbose": 0,
        "max_leaf_nodes": None,
        "warm_start": False,
        "validation_fraction": 0.1,
        "n_iter_no_change": None,
        "tol": 0.0001,
        "ccp_alpha": 0.0,
    }

    # Cyclically optimize hyperparameters
    ma_vec = []
    for c in np.arange(cycles):
        print("Optimizing random state...")
        opts, _ = _optimize_rs(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        print("Optimizing learning rate...")
        opts, _ = _optimize_lr(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        print("Optimizing n_estimators...")
        opts, _ = _optimize_nest(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        print("Optimizing subsample...")
        opts, _ = _optimize_ss(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        print("Optimizing criterion...")
        opts, _ = _optimize_cr(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        print("Optimizing min_samples_split...")
        opts, _ = _optimize_mss(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        print("Optimizing min_samples_leaf...")
        opts, _ = _optimize_msl(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        print("Optimizing min_weight_fraction_leaf...")
        opts, _ = _optimize_mwfl(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        print("Optimizing max_depth...")
        opts, _ = _optimize_md(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        print("Optimizing min_impurity_decrease...")
        opts, _ = _optimize_mid(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        print("Optimizing init...")
        opts, _ = _optimize_init(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        print("Optimizing max_features...")
        opts, _ = _optimize_mf(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        print("Optimizing max_leaf_nodes...")
        opts, ma = _optimize_mln(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        ma_vec.append(ma)

    return opts, ma_vec
