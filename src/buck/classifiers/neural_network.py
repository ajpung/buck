from typing import Any

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score


def _optimize_rs(X_train_pca, y_train_flat, X_test_pca, y_true, opts):
    # Initialize variables
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = np.arange(150)
    best_val = variable_array[0]
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = MLPClassifier(
            random_state=v,
            hidden_layer_sizes=opts["hidden_layer_sizes"],
            activation=opts["activation"],
            solver=opts["solver"],
            alpha=opts["alpha"],
            batch_size=opts["batch_size"],
            learning_rate=opts["learning_rate"],
            learning_rate_init=opts["learning_rate_init"],
            power_t=opts["power_t"],
            max_iter=opts["max_iter"],
            shuffle=opts["shuffle"],
            tol=opts["tol"],
            verbose=opts["verbose"],
            warm_start=opts["warm_start"],
            momentum=opts["momentum"],
            nesterovs_momentum=opts["nesterovs_momentum"],
            early_stopping=opts["early_stopping"],
            validation_fraction=opts["validation_fraction"],
            beta_1=opts["beta_1"],
            beta_2=opts["beta_2"],
            epsilon=opts["epsilon"],
            n_iter_no_change=opts["n_iter_no_change"],
            max_fun=opts["max_fun"],
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


def _optimize_hl(X_train_pca, y_train_flat, X_test_pca, y_true, opts):
    # Initialize variables
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = np.arange(1, 150, 1)
    best_val = variable_array[0]
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = MLPClassifier(
            random_state=opts["random_state"],
            hidden_layer_sizes=(v,),
            activation=opts["activation"],
            solver=opts["solver"],
            alpha=opts["alpha"],
            batch_size=opts["batch_size"],
            learning_rate=opts["learning_rate"],
            learning_rate_init=opts["learning_rate_init"],
            power_t=opts["power_t"],
            max_iter=opts["max_iter"],
            shuffle=opts["shuffle"],
            tol=opts["tol"],
            verbose=opts["verbose"],
            warm_start=opts["warm_start"],
            momentum=opts["momentum"],
            nesterovs_momentum=opts["nesterovs_momentum"],
            early_stopping=opts["early_stopping"],
            validation_fraction=opts["validation_fraction"],
            beta_1=opts["beta_1"],
            beta_2=opts["beta_2"],
            epsilon=opts["epsilon"],
            n_iter_no_change=opts["n_iter_no_change"],
            max_fun=opts["max_fun"],
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
    opts["hidden_layer_sizes"] = (best_val,)

    return opts, max_acc, f1s


def _optimize_act(X_train_pca, y_train_flat, X_test_pca, y_true, opts):
    # Initialize variables
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = ["identity", "logistic", "tanh", "relu"]
    best_val = variable_array[0]
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = MLPClassifier(
            random_state=opts["random_state"],
            hidden_layer_sizes=opts["hidden_layer_sizes"],
            activation=v,
            solver=opts["solver"],
            alpha=opts["alpha"],
            batch_size=opts["batch_size"],
            learning_rate=opts["learning_rate"],
            learning_rate_init=opts["learning_rate_init"],
            power_t=opts["power_t"],
            max_iter=opts["max_iter"],
            shuffle=opts["shuffle"],
            tol=opts["tol"],
            verbose=opts["verbose"],
            warm_start=opts["warm_start"],
            momentum=opts["momentum"],
            nesterovs_momentum=opts["nesterovs_momentum"],
            early_stopping=opts["early_stopping"],
            validation_fraction=opts["validation_fraction"],
            beta_1=opts["beta_1"],
            beta_2=opts["beta_2"],
            epsilon=opts["epsilon"],
            n_iter_no_change=opts["n_iter_no_change"],
            max_fun=opts["max_fun"],
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
    opts["activation"] = best_val

    return opts, max_acc, f1s


def _optimize_solver(X_train_pca, y_train_flat, X_test_pca, y_true, opts):
    # Initialize variables
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = ["lbfgs", "sgd", "adam"]
    best_val = variable_array[0]
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = MLPClassifier(
            random_state=opts["random_state"],
            hidden_layer_sizes=opts["hidden_layer_sizes"],
            activation=opts["activation"],
            solver=v,
            alpha=opts["alpha"],
            batch_size=opts["batch_size"],
            learning_rate=opts["learning_rate"],
            learning_rate_init=opts["learning_rate_init"],
            power_t=opts["power_t"],
            max_iter=opts["max_iter"],
            shuffle=opts["shuffle"],
            tol=opts["tol"],
            verbose=opts["verbose"],
            warm_start=opts["warm_start"],
            momentum=opts["momentum"],
            nesterovs_momentum=opts["nesterovs_momentum"],
            early_stopping=opts["early_stopping"],
            validation_fraction=opts["validation_fraction"],
            beta_1=opts["beta_1"],
            beta_2=opts["beta_2"],
            epsilon=opts["epsilon"],
            n_iter_no_change=opts["n_iter_no_change"],
            max_fun=opts["max_fun"],
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
    opts["solver"] = best_val

    return opts, max_acc, f1s


def _optimize_alpha(X_train_pca, y_train_flat, X_test_pca, y_true, opts):
    # Initialize variables
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = np.arange(0.0001, 0.1, 0.0001)
    best_val = variable_array[0]
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = MLPClassifier(
            random_state=opts["random_state"],
            hidden_layer_sizes=opts["hidden_layer_sizes"],
            activation=opts["activation"],
            solver=opts["solver"],
            alpha=v,
            batch_size=opts["batch_size"],
            learning_rate=opts["learning_rate"],
            learning_rate_init=opts["learning_rate_init"],
            power_t=opts["power_t"],
            max_iter=opts["max_iter"],
            shuffle=opts["shuffle"],
            tol=opts["tol"],
            verbose=opts["verbose"],
            warm_start=opts["warm_start"],
            momentum=opts["momentum"],
            nesterovs_momentum=opts["nesterovs_momentum"],
            early_stopping=opts["early_stopping"],
            validation_fraction=opts["validation_fraction"],
            beta_1=opts["beta_1"],
            beta_2=opts["beta_2"],
            epsilon=opts["epsilon"],
            n_iter_no_change=opts["n_iter_no_change"],
            max_fun=opts["max_fun"],
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
    opts["alpha"] = best_val

    return opts, max_acc, f1s


def _optimize_batch(X_train_pca, y_train_flat, X_test_pca, y_true, opts):
    # Initialize variables
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = np.arange(1, 100, 1)
    variable_array = np.append(variable_array.astype(object), "auto")
    best_val = variable_array[0]
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = MLPClassifier(
            random_state=opts["random_state"],
            hidden_layer_sizes=opts["hidden_layer_sizes"],
            activation=opts["activation"],
            solver=opts["solver"],
            alpha=opts["alpha"],
            batch_size=v,
            learning_rate=opts["learning_rate"],
            learning_rate_init=opts["learning_rate_init"],
            power_t=opts["power_t"],
            max_iter=opts["max_iter"],
            shuffle=opts["shuffle"],
            tol=opts["tol"],
            verbose=opts["verbose"],
            warm_start=opts["warm_start"],
            momentum=opts["momentum"],
            nesterovs_momentum=opts["nesterovs_momentum"],
            early_stopping=opts["early_stopping"],
            validation_fraction=opts["validation_fraction"],
            beta_1=opts["beta_1"],
            beta_2=opts["beta_2"],
            epsilon=opts["epsilon"],
            n_iter_no_change=opts["n_iter_no_change"],
            max_fun=opts["max_fun"],
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
    opts["batch_size"] = best_val

    return opts, max_acc, f1s


def _optimize_lr(X_train_pca, y_train_flat, X_test_pca, y_true, opts):
    # Initialize variables
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = np.arange(0.0001, 0.1, 0.0001)
    best_val = variable_array[0]
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = MLPClassifier(
            random_state=opts["random_state"],
            hidden_layer_sizes=opts["hidden_layer_sizes"],
            activation=opts["activation"],
            solver=opts["solver"],
            alpha=opts["alpha"],
            batch_size=opts["batch_size"],
            learning_rate=v,
            learning_rate_init=opts["learning_rate_init"],
            power_t=opts["power_t"],
            max_iter=opts["max_iter"],
            shuffle=opts["shuffle"],
            tol=opts["tol"],
            verbose=opts["verbose"],
            warm_start=opts["warm_start"],
            momentum=opts["momentum"],
            nesterovs_momentum=opts["nesterovs_momentum"],
            early_stopping=opts["early_stopping"],
            validation_fraction=opts["validation_fraction"],
            beta_1=opts["beta_1"],
            beta_2=opts["beta_2"],
            epsilon=opts["epsilon"],
            n_iter_no_change=opts["n_iter_no_change"],
            max_fun=opts["max_fun"],
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
    opts["learning_rate"] = best_val

    return opts, max_acc, f1s


def _optimize_power(X_train_pca, y_train_flat, X_test_pca, y_true, opts):
    # Initialize variables
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = np.arange(0.01, 0.5, 0.01)
    best_val = variable_array[0]
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = MLPClassifier(
            random_state=opts["random_state"],
            hidden_layer_sizes=opts["hidden_layer_sizes"],
            activation=opts["activation"],
            solver=opts["solver"],
            alpha=opts["alpha"],
            batch_size=opts["batch_size"],
            learning_rate=opts["learning_rate"],
            learning_rate_init=opts["learning_rate_init"],
            power_t=v,
            max_iter=opts["max_iter"],
            shuffle=opts["shuffle"],
            tol=opts["tol"],
            verbose=opts["verbose"],
            warm_start=opts["warm_start"],
            momentum=opts["momentum"],
            nesterovs_momentum=opts["nesterovs_momentum"],
            early_stopping=opts["early_stopping"],
            validation_fraction=opts["validation_fraction"],
            beta_1=opts["beta_1"],
            beta_2=opts["beta_2"],
            epsilon=opts["epsilon"],
            n_iter_no_change=opts["n_iter_no_change"],
            max_fun=opts["max_fun"],
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
    opts["power_t"] = best_val

    return opts, max_acc, f1s


def _optimize_mi(X_train_pca, y_train_flat, X_test_pca, y_true, opts):
    # Initialize variables
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = np.arange(1, 100, 1)
    best_val = variable_array[0]
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = MLPClassifier(
            random_state=opts["random_state"],
            hidden_layer_sizes=opts["hidden_layer_sizes"],
            activation=opts["activation"],
            solver=opts["solver"],
            alpha=opts["alpha"],
            batch_size=opts["batch_size"],
            learning_rate=opts["learning_rate"],
            learning_rate_init=opts["learning_rate_init"],
            power_t=opts["power_t"],
            max_iter=v,
            shuffle=opts["shuffle"],
            tol=opts["tol"],
            verbose=opts["verbose"],
            warm_start=opts["warm_start"],
            momentum=opts["momentum"],
            nesterovs_momentum=opts["nesterovs_momentum"],
            early_stopping=opts["early_stopping"],
            validation_fraction=opts["validation_fraction"],
            beta_1=opts["beta_1"],
            beta_2=opts["beta_2"],
            epsilon=opts["epsilon"],
            n_iter_no_change=opts["n_iter_no_change"],
            max_fun=opts["max_fun"],
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


def _optimize_tol(X_train_pca, y_train_flat, X_test_pca, y_true, opts):
    # Initialize variables
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = np.arange(0.0001, 0.1, 0.0001)
    best_val = variable_array[0]
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = MLPClassifier(
            random_state=opts["random_state"],
            hidden_layer_sizes=opts["hidden_layer_sizes"],
            activation=opts["activation"],
            solver=opts["solver"],
            alpha=opts["alpha"],
            batch_size=opts["batch_size"],
            learning_rate=opts["learning_rate"],
            learning_rate_init=opts["learning_rate_init"],
            power_t=opts["power_t"],
            max_iter=opts["max_iter"],
            shuffle=opts["shuffle"],
            tol=v,
            verbose=opts["verbose"],
            warm_start=opts["warm_start"],
            momentum=opts["momentum"],
            nesterovs_momentum=opts["nesterovs_momentum"],
            early_stopping=opts["early_stopping"],
            validation_fraction=opts["validation_fraction"],
            beta_1=opts["beta_1"],
            beta_2=opts["beta_2"],
            epsilon=opts["epsilon"],
            n_iter_no_change=opts["n_iter_no_change"],
            max_fun=opts["max_fun"],
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


def _optimize_shuffle(X_train_pca, y_train_flat, X_test_pca, y_true, opts):
    # Initialize variables
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = [True, False]
    best_val = variable_array[0]
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = MLPClassifier(
            random_state=opts["random_state"],
            hidden_layer_sizes=opts["hidden_layer_sizes"],
            activation=opts["activation"],
            solver=opts["solver"],
            alpha=opts["alpha"],
            batch_size=opts["batch_size"],
            learning_rate=opts["learning_rate"],
            learning_rate_init=opts["learning_rate_init"],
            power_t=opts["power_t"],
            max_iter=opts["max_iter"],
            shuffle=v,
            tol=opts["tol"],
            verbose=opts["verbose"],
            warm_start=opts["warm_start"],
            momentum=opts["momentum"],
            nesterovs_momentum=opts["nesterovs_momentum"],
            early_stopping=opts["early_stopping"],
            validation_fraction=opts["validation_fraction"],
            beta_1=opts["beta_1"],
            beta_2=opts["beta_2"],
            epsilon=opts["epsilon"],
            n_iter_no_change=opts["n_iter_no_change"],
            max_fun=opts["max_fun"],
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


def _optimize_momentum(X_train_pca, y_train_flat, X_test_pca, y_true, opts):
    # Initialize variables
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = np.arange(0.1, 1.0, 0.01)
    best_val = variable_array[0]
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = MLPClassifier(
            random_state=opts["random_state"],
            hidden_layer_sizes=opts["hidden_layer_sizes"],
            activation=opts["activation"],
            solver=opts["solver"],
            alpha=opts["alpha"],
            batch_size=opts["batch_size"],
            learning_rate=opts["learning_rate"],
            learning_rate_init=opts["learning_rate_init"],
            power_t=opts["power_t"],
            max_iter=opts["max_iter"],
            shuffle=opts["shuffle"],
            tol=opts["tol"],
            verbose=opts["verbose"],
            warm_start=opts["warm_start"],
            momentum=v,
            nesterovs_momentum=opts["nesterovs_momentum"],
            early_stopping=opts["early_stopping"],
            validation_fraction=opts["validation_fraction"],
            beta_1=opts["beta_1"],
            beta_2=opts["beta_2"],
            epsilon=opts["epsilon"],
            n_iter_no_change=opts["n_iter_no_change"],
            max_fun=opts["max_fun"],
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
    opts["momentum"] = best_val

    return opts, max_acc, f1s


def _optimize_nesterovs(X_train_pca, y_train_flat, X_test_pca, y_true, opts):
    # Initialize variables
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = [True, False]
    best_val = variable_array[0]
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = MLPClassifier(
            random_state=opts["random_state"],
            hidden_layer_sizes=opts["hidden_layer_sizes"],
            activation=opts["activation"],
            solver=opts["solver"],
            alpha=opts["alpha"],
            batch_size=opts["batch_size"],
            learning_rate=opts["learning_rate"],
            learning_rate_init=opts["learning_rate_init"],
            power_t=opts["power_t"],
            max_iter=opts["max_iter"],
            shuffle=opts["shuffle"],
            tol=opts["tol"],
            verbose=opts["verbose"],
            warm_start=opts["warm_start"],
            momentum=opts["momentum"],
            nesterovs_momentum=v,
            early_stopping=opts["early_stopping"],
            validation_fraction=opts["validation_fraction"],
            beta_1=opts["beta_1"],
            beta_2=opts["beta_2"],
            epsilon=opts["epsilon"],
            n_iter_no_change=opts["n_iter_no_change"],
            max_fun=opts["max_fun"],
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
    opts["nesterovs_momentum"] = best_val

    return opts, max_acc, f1s


def _optimize_vf(X_train_pca, y_train_flat, X_test_pca, y_true, opts):
    # Initialize variables
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = np.arange(0.01, 1.0, 0.01)
    best_val = variable_array[0]
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = MLPClassifier(
            random_state=opts["random_state"],
            hidden_layer_sizes=opts["hidden_layer_sizes"],
            activation=opts["activation"],
            solver=opts["solver"],
            alpha=opts["alpha"],
            batch_size=opts["batch_size"],
            learning_rate=opts["learning_rate"],
            learning_rate_init=opts["learning_rate_init"],
            power_t=opts["power_t"],
            max_iter=opts["max_iter"],
            shuffle=opts["shuffle"],
            tol=opts["tol"],
            verbose=opts["verbose"],
            warm_start=opts["warm_start"],
            momentum=opts["momentum"],
            nesterovs_momentum=opts["nesterovs_momentum"],
            early_stopping=opts["early_stopping"],
            validation_fraction=v,
            beta_1=opts["beta_1"],
            beta_2=opts["beta_2"],
            epsilon=opts["epsilon"],
            n_iter_no_change=opts["n_iter_no_change"],
            max_fun=opts["max_fun"],
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


def _optimize_beta1(X_train_pca, y_train_flat, X_test_pca, y_true, opts):
    # Initialize variables
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = np.arange(0.0, 0.999, 0.001)
    best_val = variable_array[0]
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = MLPClassifier(
            random_state=opts["random_state"],
            hidden_layer_sizes=opts["hidden_layer_sizes"],
            activation=opts["activation"],
            solver=opts["solver"],
            alpha=opts["alpha"],
            batch_size=opts["batch_size"],
            learning_rate=opts["learning_rate"],
            learning_rate_init=opts["learning_rate_init"],
            power_t=opts["power_t"],
            max_iter=opts["max_iter"],
            shuffle=opts["shuffle"],
            tol=opts["tol"],
            verbose=opts["verbose"],
            warm_start=opts["warm_start"],
            momentum=opts["momentum"],
            nesterovs_momentum=opts["nesterovs_momentum"],
            early_stopping=opts["early_stopping"],
            validation_fraction=opts["validation_fraction"],
            beta_1=v,
            beta_2=opts["beta_2"],
            epsilon=opts["epsilon"],
            n_iter_no_change=opts["n_iter_no_change"],
            max_fun=opts["max_fun"],
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
    opts["beta_1"] = best_val

    return opts, max_acc, f1s


def _optimize_beta2(X_train_pca, y_train_flat, X_test_pca, y_true, opts):
    # Initialize variables
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = np.arange(0.0, 0.999, 0.001)
    best_val = variable_array[0]
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = MLPClassifier(
            random_state=opts["random_state"],
            hidden_layer_sizes=opts["hidden_layer_sizes"],
            activation=opts["activation"],
            solver=opts["solver"],
            alpha=opts["alpha"],
            batch_size=opts["batch_size"],
            learning_rate=opts["learning_rate"],
            learning_rate_init=opts["learning_rate_init"],
            power_t=opts["power_t"],
            max_iter=opts["max_iter"],
            shuffle=opts["shuffle"],
            tol=opts["tol"],
            verbose=opts["verbose"],
            warm_start=opts["warm_start"],
            momentum=opts["momentum"],
            nesterovs_momentum=opts["nesterovs_momentum"],
            early_stopping=opts["early_stopping"],
            validation_fraction=opts["validation_fraction"],
            beta_1=opts["beta_1"],
            beta_2=v,
            epsilon=opts["epsilon"],
            n_iter_no_change=opts["n_iter_no_change"],
            max_fun=opts["max_fun"],
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
    opts["beta_2"] = best_val

    return opts, max_acc, f1s


def _optimize_eps(X_train_pca, y_train_flat, X_test_pca, y_true, opts):
    # Initialize variables
    ac_vec = []
    f1_vec = []
    max_acc = -np.inf
    max_idx = -1
    variable_array = np.arange(1e-10, 1e-5, 1e-10)
    best_val = variable_array[0]
    for i in np.arange(len(variable_array)):
        v = variable_array[i]
        # Define classifiers to test
        classifier = MLPClassifier(
            random_state=opts["random_state"],
            hidden_layer_sizes=opts["hidden_layer_sizes"],
            activation=opts["activation"],
            solver=opts["solver"],
            alpha=opts["alpha"],
            batch_size=opts["batch_size"],
            learning_rate=opts["learning_rate"],
            learning_rate_init=opts["learning_rate_init"],
            power_t=opts["power_t"],
            max_iter=opts["max_iter"],
            shuffle=opts["shuffle"],
            tol=opts["tol"],
            verbose=opts["verbose"],
            warm_start=opts["warm_start"],
            momentum=opts["momentum"],
            nesterovs_momentum=opts["nesterovs_momentum"],
            early_stopping=opts["early_stopping"],
            validation_fraction=opts["validation_fraction"],
            beta_1=opts["beta_1"],
            beta_2=opts["beta_2"],
            epsilon=v,
            n_iter_no_change=opts["n_iter_no_change"],
            max_fun=opts["max_fun"],
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
    opts["epsilon"] = best_val

    return opts, max_acc, f1s


def _optimize_neural_network(X_train_pca, y_train_flat, X_test_pca, y_true, cycles=2):

    # Shorten parameters
    Xtr_pca = X_train_pca
    ytr_flat = y_train_flat
    Xte_pca = X_test_pca

    # Define optimals
    opts = {
        "hidden_layer_sizes": (100,),
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.0001,
        "batch_size": "auto",
        "learning_rate": "constant",
        "learning_rate_init": 0.001,
        "power_t": 0.5,
        "max_iter": 20000,
        "shuffle": True,
        "random_state": None,
        "tol": 0.01,
        "verbose": False,
        "warm_start": False,
        "momentum": 0.9,
        "nesterovs_momentum": True,
        "early_stopping": False,
        "validation_fraction": 0.1,
        "beta_1": 0.9,
        "beta_2": 0.999,
        "epsilon": 1e-08,
        "n_iter_no_change": 10,
        "max_fun": 15000,
    }

    # Cyclically optimize hyperparameters
    ma_vec = []
    f1_vec = []
    for c in np.arange(cycles):
        opts, _, _ = _optimize_rs(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _, _ = _optimize_hl(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _, _ = _optimize_act(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _, _ = _optimize_solver(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _, _ = _optimize_alpha(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _, _ = _optimize_batch(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _, _ = _optimize_lr(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _, _ = _optimize_power(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _, _ = _optimize_shuffle(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _, _ = _optimize_momentum(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _, _ = _optimize_nesterovs(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _, _ = _optimize_vf(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _, _ = _optimize_beta1(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, _, _ = _optimize_beta2(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        opts, ma, f1 = _optimize_eps(Xtr_pca, ytr_flat, Xte_pca, y_true, opts)
        ma_vec.append(ma)
        f1_vec.append(f1)

    return opts, ma, f1, ma_vec, f1_vec
