import gc

from buck.classifiers.ada_boost import _optimize_ada_boost
from buck.classifiers.bagging_classifier import _optimize_bagging
from buck.classifiers.decision_tree import _optimize_decision_tree
from buck.classifiers.extra_trees import _optimize_extra_trees
from buck.classifiers.gaussian_process import _optimize_gaussian_process
from buck.classifiers.gradient_boost import _optimize_gradient_boost
from buck.classifiers.k_nearest import _optimize_knn
from buck.classifiers.linear_discriminant import _optimize_linear_discriminant
from buck.classifiers.logistic_regression import _optimize_logistic_regression
from buck.classifiers.naive_bayes import _optimize_naive_bayes
from buck.classifiers.neural_network import _optimize_neural_network
from buck.classifiers.passive_aggressive import _optimize_passive_aggressive
from buck.classifiers.random_forest import _optimize_random_forest
from buck.classifiers.ridge_classifier import _optimize_ridge
from buck.classifiers.self_training import _optimize_self_training
from buck.classifiers.stacking_classifier import _optimize_stacking_classifier
from buck.classifiers.stocastic_gradient_descent import _optimize_sgd_classifier
from buck.classifiers.voting_classifier import _optimize_voting_classifier

"""
To validate:
- Neural Network
"""


def garbage_collect():
    # Force garbage collection
    gc.collect()


def write_to_nested_dict(data, keys, value):
    current = data
    for key in keys[:-1]:
        current = current.setdefault(key, {})
    current[keys[-1]] = value


def optimize_all(X_train, y_train, X_test, y_true, cycles=2):
    results = {}
    print("Optimizing")
    # ------------------------ ADABOOST -----------------------
    print("...adaboost")
    opts, ma, f1, ma_vec, f1_vec = _optimize_ada_boost(
        X_train, y_train, X_test, y_true, cycles=cycles
    )
    write_to_nested_dict(results, ["Adaboost", "Accuracy"], ma)
    write_to_nested_dict(results, ["Adaboost", "f1-score"], f1)
    write_to_nested_dict(results, ["Adaboost", "optimals"], opts)
    print(results, "\n")
    gc.collect()
    # ------------------- BAGGING CLASSIFIER ------------------
    print("...bagging classifier")
    opts, ma, f1, ma_vec, f1_vec = _optimize_bagging(
        X_train, y_train, X_test, y_true, cycles=cycles
    )
    write_to_nested_dict(results, ["Bagging Classifier", "Accuracy"], ma)
    write_to_nested_dict(results, ["Bagging Classifier", "f1-score"], f1)
    write_to_nested_dict(results, ["Bagging Classifier", "optimals"], opts)
    print(results, "\n")
    gc.collect()
    # -------------------- DECISION TREES ---------------------
    print("...decision tree")
    opts, ma, f1, ma_vec, f1_vec = _optimize_decision_tree(
        X_train, y_train, X_test, y_true, cycles=cycles
    )
    write_to_nested_dict(results, ["Decision Tree", "Accuracy"], ma)
    write_to_nested_dict(results, ["Decision Tree", "f1-score"], f1)
    write_to_nested_dict(results, ["Decision Tree", "optimals"], opts)
    print(results, "\n")
    gc.collect()
    # ---------------------- EXTRA TREES ----------------------
    print("...extra trees")
    opts, ma, f1, ma_vec, f1_vec = _optimize_extra_trees(
        X_train, y_train, X_test, y_true, cycles=cycles
    )
    write_to_nested_dict(results, ["Extra Trees", "Accuracy"], ma)
    write_to_nested_dict(results, ["Extra Trees", "f1-score"], f1)
    write_to_nested_dict(results, ["Extra Trees", "optimals"], opts)
    print(results, "\n")
    gc.collect()
    # -------------------- GRADIENT BOOST ---------------------
    print("...gradient boost")
    opts, ma, f1, ma_vec, f1_vec = _optimize_gradient_boost(
        X_train, y_train, X_test, y_true, cycles=cycles
    )
    write_to_nested_dict(results, ["Gradient Boost", "Accuracy"], ma)
    write_to_nested_dict(results, ["Gradient Boost", "f1-score"], f1)
    write_to_nested_dict(results, ["Gradient Boost", "optimals"], opts)
    print(results, "\n")
    gc.collect()
    return results
    # --------------------- RANDOM FOREST ---------------------
    print("...random forest")
    print(results, "\n")
    opts, ma, f1, ma_vec, f1_vec = _optimize_random_forest(
        X_train, y_train, X_test, y_true, cycles=cycles
    )
    write_to_nested_dict(results, ["Random Forest", "Accuracy"], ma)
    write_to_nested_dict(results, ["Random Forest", "f1-score"], f1)
    write_to_nested_dict(results, ["Random Forest", "optimals"], opts)
    print(results, "\n")
    gc.collect()
    # ------------------ LINEAR DISCRIMINANT ------------------
    print("...linear discriminant")
    opts, ma, f1, ma_vec, f1_vec = _optimize_linear_discriminant(
        X_train, y_train, X_test, y_true, cycles=cycles
    )
    write_to_nested_dict(results, ["Linear Discriminant", "Accuracy"], ma)
    write_to_nested_dict(results, ["Linear Discriminant", "f1-score"], f1)
    write_to_nested_dict(results, ["Linear Discriminant", "optimals"], opts)
    print(results, "\n")
    gc.collect()
    # ------------------ LOGISTIC REGRESSION ------------------
    print("...logistic regression")
    opts, ma, f1, ma_vec, f1_vec = _optimize_logistic_regression(
        X_train, y_train, X_test, y_true, cycles=cycles
    )
    write_to_nested_dict(results, ["Logistic Regression", "Accuracy"], ma)
    write_to_nested_dict(results, ["Logistic Regression", "f1-score"], f1)
    write_to_nested_dict(results, ["Logistic Regression", "optimals"], opts)
    print(results, "\n")
    gc.collect()
    # ------------------ PASSIVE AGGRESSIVE -------------------
    print("...passive aggressive")
    opts, ma, f1, ma_vec, f1_vec = _optimize_passive_aggressive(
        X_train, y_train, X_test, y_true, cycles=cycles
    )
    write_to_nested_dict(results, ["Passive Aggressive", "Accuracy"], ma)
    write_to_nested_dict(results, ["Passive Aggressive", "f1-score"], f1)
    write_to_nested_dict(results, ["Passive Aggressive", "optimals"], opts)
    print(results, "\n")
    gc.collect()
    # ------------------------ RIDGE --------------------------
    print("...ridge")
    opts, ma, f1, ma_vec, f1_vec = _optimize_ridge(
        X_train, y_train, X_test, y_true, cycles=cycles
    )
    write_to_nested_dict(results, ["Ridge", "Accuracy"], ma)
    write_to_nested_dict(results, ["Ridge", "f1-score"], f1)
    write_to_nested_dict(results, ["Ridge", "optimals"], opts)
    print(results, "\n")
    gc.collect()
    # ------------------------ VOTING -------------------------
    print("...voting")
    opts, ma, f1, ma_vec, f1_vec = _optimize_voting_classifier(
        X_train, y_train, X_test, y_true, cycles=cycles
    )
    write_to_nested_dict(results, ["Voting Classifier", "Accuracy"], ma)
    write_to_nested_dict(results, ["Voting Classifier", "f1-score"], f1)
    write_to_nested_dict(results, ["Voting Classifier", "optimals"], opts)
    print(results, "\n")
    gc.collect()
    # --------------------- SELF-TRAINING ---------------------
    print("...self-training")
    opts, ma, f1, ma_vec, f1_vec = _optimize_self_training(
        X_train, y_train, X_test, y_true, cycles=cycles
    )
    write_to_nested_dict(results, ["Naive Bayes", "Accuracy"], ma)
    write_to_nested_dict(results, ["Naive Bayes", "f1-score"], f1)
    write_to_nested_dict(results, ["Naive Bayes", "optimals"], opts)
    print(results, "\n")
    gc.collect()
    # -------------- STOCHASTIC GRADIENT DESCENT --------------
    print("...stochastic gradient descent")
    opts, ma, f1, ma_vec, f1_vec = _optimize_sgd_classifier(
        X_train, y_train, X_test, y_true, cycles=cycles
    )
    write_to_nested_dict(results, ["Stochastic Gradient Descent", "Accuracy"], ma)
    write_to_nested_dict(results, ["Stochastic Gradient Descent", "f1-score"], f1)
    write_to_nested_dict(results, ["Stochastic Gradient Descent", "optimals"], opts)
    print(results, "\n")
    gc.collect()
    # ------------------------ STACKING -----------------------
    print("...stacking")
    opts, ma, f1, ma_vec, f1_vec = _optimize_stacking_classifier(
        X_train, y_train, X_test, y_true, cycles=cycles
    )
    write_to_nested_dict(results, ["Stacking Classifier", "Accuracy"], ma)
    write_to_nested_dict(results, ["Stacking Classifier", "f1-score"], f1)
    write_to_nested_dict(results, ["Stacking Classifier", "optimals"], opts)
    print(results, "\n")
    gc.collect()
    # ---------------------- NAIVE BAYES ----------------------
    print("...naive bayes")
    opts, ma, f1, ma_vec, f1_vec = _optimize_naive_bayes(
        X_train, y_train, X_test, y_true, cycles=cycles
    )
    write_to_nested_dict(results, ["Naive Bayes", "Accuracy"], ma)
    write_to_nested_dict(results, ["Naive Bayes", "f1-score"], f1)
    write_to_nested_dict(results, ["Naive Bayes", "optimals"], opts)
    print(results, "\n")
    gc.collect()
    # -------------------------- KNN --------------------------
    print("...KNN")
    opts, ma, f1, ma_vec, f1_vec = _optimize_knn(
        X_train, y_train, X_test, y_true, cycles=cycles
    )
    write_to_nested_dict(results, ["KNN", "Accuracy"], ma)
    write_to_nested_dict(results, ["KNN", "f1-score"], f1)
    write_to_nested_dict(results, ["KNN", "optimals"], opts)
    print(results, "\n")
    gc.collect()
    # --------------------- NEURAL NETWORK --------------------
    print("...multi-layer perceptron")
    opts, ma, f1, ma_vec, f1_vec = _optimize_neural_network(
        X_train, y_train, X_test, y_true, cycles=cycles
    )
    write_to_nested_dict(results, ["Neural Network", "Accuracy"], ma)
    write_to_nested_dict(results, ["Neural Network", "f1-score"], f1)
    write_to_nested_dict(results, ["Neural Network", "optimals"], opts)
    print(results, "\n")
    gc.collect()

    # ------------------- GAUSSIAN PROCESS --------------------
    print("...Gaussian Process")
    opts, ma, f1, ma_vec, f1_vec = _optimize_gaussian_process(
        X_train, y_train, X_test, y_true, cycles=cycles
    )
    write_to_nested_dict(results, ["Gaussian Process", "Accuracy"], ma)
    write_to_nested_dict(results, ["Gaussian Process", "f1-score"], f1)
    write_to_nested_dict(results, ["Gaussian Process", "optimals"], opts)
    print(results, "\n")
    gc.collect()
