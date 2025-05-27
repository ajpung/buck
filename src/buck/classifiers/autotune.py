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
    ## ------------------------ ADABOOST -----------------------
    # print("\n...adaboost")
    # opts, ma, f1, ma_vec, f1_vec = _optimize_ada_boost(
    #    X_train, y_train, X_test, y_true, cycles=cycles
    # )
    # write_to_nested_dict(results, ["Adaboost", "Accuracy"], ma)
    # write_to_nested_dict(results, ["Adaboost", "f1-score"], f1)
    # write_to_nested_dict(results, ["Adaboost", "optimals"], opts)
    # print(results["Adaboost"], "\n")
    # gc.collect()
    ## ------------------- BAGGING CLASSIFIER ------------------
    # print("\n...bagging classifier")
    # opts, ma, f1, ma_vec, f1_vec = _optimize_bagging(
    #    X_train, y_train, X_test, y_true, cycles=cycles
    # )
    # write_to_nested_dict(results, ["Bagging Classifier", "Accuracy"], ma)
    # write_to_nested_dict(results, ["Bagging Classifier", "f1-score"], f1)
    # write_to_nested_dict(results, ["Bagging Classifier", "optimals"], opts)
    # print(results["Bagging Classifier"], "\n")
    # gc.collect()
    # -------------------- DECISION TREES ---------------------
    print("\n...decision tree")
    opts, ma, f1, ma_vec, f1_vec = _optimize_decision_tree(
        X_train, y_train, X_test, y_true, cycles=cycles
    )
    write_to_nested_dict(results, ["Decision Tree", "Accuracy"], ma)
    write_to_nested_dict(results, ["Decision Tree", "f1-score"], f1)
    write_to_nested_dict(results, ["Decision Tree", "optimals"], opts)
    print(results["Decision Tree"], "\n")
    gc.collect()
    """
    # ---------------------- EXTRA TREES ----------------------
    print("\n...extra trees")
    opts, ma, f1, ma_vec, f1_vec = _optimize_extra_trees(
        X_train, y_train, X_test, y_true, cycles=cycles
    )
    write_to_nested_dict(results, ["Extra Trees", "Accuracy"], ma)
    write_to_nested_dict(results, ["Extra Trees", "f1-score"], f1)
    write_to_nested_dict(results, ["Extra Trees", "optimals"], opts)
    print(results["Extra Trees"], "\n")
    gc.collect()
    # -------------------- GRADIENT BOOST ---------------------
    print("\n...gradient boost")
    opts, ma, f1, ma_vec, f1_vec = _optimize_gradient_boost(
        X_train, y_train, X_test, y_true, cycles=cycles
    )
    write_to_nested_dict(results, ["Gradient Boost", "Accuracy"], ma)
    write_to_nested_dict(results, ["Gradient Boost", "f1-score"], f1)
    write_to_nested_dict(results, ["Gradient Boost", "optimals"], opts)
    print(results["Gradient Boost"], "\n")
    gc.collect()
    return results
    # --------------------- RANDOM FOREST ---------------------
    print("\n...random forest")
    opts, ma, f1, ma_vec, f1_vec = _optimize_random_forest(
        X_train, y_train, X_test, y_true, cycles=cycles
    )
    write_to_nested_dict(results, ["Random Forest", "Accuracy"], ma)
    write_to_nested_dict(results, ["Random Forest", "f1-score"], f1)
    write_to_nested_dict(results, ["Random Forest", "optimals"], opts)
    print(results["Random Forest"], "\n")
    gc.collect()
    # ------------------ LINEAR DISCRIMINANT ------------------
    print("\n...linear discriminant")
    opts, ma, f1, ma_vec, f1_vec = _optimize_linear_discriminant(
        X_train, y_train, X_test, y_true, cycles=cycles
    )
    write_to_nested_dict(results, ["Linear Discriminant", "Accuracy"], ma)
    write_to_nested_dict(results, ["Linear Discriminant", "f1-score"], f1)
    write_to_nested_dict(results, ["Linear Discriminant", "optimals"], opts)
    print(results["Linear Discriminant"], "\n")
    gc.collect()
    # ------------------ LOGISTIC REGRESSION ------------------
    print("\n...logistic regression")
    opts, ma, f1, ma_vec, f1_vec = _optimize_logistic_regression(
        X_train, y_train, X_test, y_true, cycles=cycles
    )
    write_to_nested_dict(results, ["Logistic Regression", "Accuracy"], ma)
    write_to_nested_dict(results, ["Logistic Regression", "f1-score"], f1)
    write_to_nested_dict(results, ["Logistic Regression", "optimals"], opts)
    print(results["Logistic Regression"], "\n")
    gc.collect()
    # ------------------ PASSIVE AGGRESSIVE -------------------
    print("\n...passive aggressive")
    opts, ma, f1, ma_vec, f1_vec = _optimize_passive_aggressive(
        X_train, y_train, X_test, y_true, cycles=cycles
    )
    write_to_nested_dict(results, ["Passive Aggressive", "Accuracy"], ma)
    write_to_nested_dict(results, ["Passive Aggressive", "f1-score"], f1)
    write_to_nested_dict(results, ["Passive Aggressive", "optimals"], opts)
    print(results["Passive Aggressive"], "\n")
    gc.collect()
    # ------------------------ RIDGE --------------------------
    print("\n...ridge")
    opts, ma, f1, ma_vec, f1_vec = _optimize_ridge(
        X_train, y_train, X_test, y_true, cycles=cycles
    )
    write_to_nested_dict(results, ["Ridge", "Accuracy"], ma)
    write_to_nested_dict(results, ["Ridge", "f1-score"], f1)
    write_to_nested_dict(results, ["Ridge", "optimals"], opts)
    print(results["Ridge"], "\n")
    gc.collect()
    # ------------------------ VOTING -------------------------
    print("\n...voting")
    opts, ma, f1, ma_vec, f1_vec = _optimize_voting_classifier(
        X_train, y_train, X_test, y_true, cycles=cycles
    )
    write_to_nested_dict(results, ["Voting Classifier", "Accuracy"], ma)
    write_to_nested_dict(results, ["Voting Classifier", "f1-score"], f1)
    write_to_nested_dict(results, ["Voting Classifier", "optimals"], opts)
    print(results["Voting Classifier"], "\n")
    gc.collect()
    # --------------------- SELF-TRAINING ---------------------
    print("\n...self-training")
    opts, ma, f1, ma_vec, f1_vec = _optimize_self_training(
        X_train, y_train, X_test, y_true, cycles=cycles
    )
    write_to_nested_dict(results, ["Naive Bayes", "Accuracy"], ma)
    write_to_nested_dict(results, ["Naive Bayes", "f1-score"], f1)
    write_to_nested_dict(results, ["Naive Bayes", "optimals"], opts)
    print(results["Naive Bayes"], "\n")
    gc.collect()
    # -------------- STOCHASTIC GRADIENT DESCENT --------------
    print("\n...stochastic gradient descent")
    opts, ma, f1, ma_vec, f1_vec = _optimize_sgd_classifier(
        X_train, y_train, X_test, y_true, cycles=cycles
    )
    write_to_nested_dict(results, ["Stochastic Gradient Descent", "Accuracy"], ma)
    write_to_nested_dict(results, ["Stochastic Gradient Descent", "f1-score"], f1)
    write_to_nested_dict(results, ["Stochastic Gradient Descent", "optimals"], opts)
    print(results["Stochastic Gradient Descent"], "\n")
    gc.collect()
    # ------------------------ STACKING -----------------------
    print("\n...stacking")
    opts, ma, f1, ma_vec, f1_vec = _optimize_stacking_classifier(
        X_train, y_train, X_test, y_true, cycles=cycles
    )
    write_to_nested_dict(results, ["Stacking Classifier", "Accuracy"], ma)
    write_to_nested_dict(results, ["Stacking Classifier", "f1-score"], f1)
    write_to_nested_dict(results, ["Stacking Classifier", "optimals"], opts)
    print(results["Stacking Classifier"], "\n")
    gc.collect()
    # ---------------------- NAIVE BAYES ----------------------
    print("\n...naive bayes")
    opts, ma, f1, ma_vec, f1_vec = _optimize_naive_bayes(
        X_train, y_train, X_test, y_true, cycles=cycles
    )
    write_to_nested_dict(results, ["Naive Bayes", "Accuracy"], ma)
    write_to_nested_dict(results, ["Naive Bayes", "f1-score"], f1)
    write_to_nested_dict(results, ["Naive Bayes", "optimals"], opts)
    print(results["Naive Bayes"], "\n")
    gc.collect()
    # -------------------------- KNN --------------------------
    print("\n...KNN")
    opts, ma, f1, ma_vec, f1_vec = _optimize_knn(
        X_train, y_train, X_test, y_true, cycles=cycles
    )
    write_to_nested_dict(results, ["KNN", "Accuracy"], ma)
    write_to_nested_dict(results, ["KNN", "f1-score"], f1)
    write_to_nested_dict(results, ["KNN", "optimals"], opts)
    print(results["KNN"], "\n")
    gc.collect()
    # --------------------- NEURAL NETWORK --------------------
    print("\n...multi-layer perceptron")
    opts, ma, f1, ma_vec, f1_vec = _optimize_neural_network(
        X_train, y_train, X_test, y_true, cycles=cycles
    )
    write_to_nested_dict(results, ["Neural Network", "Accuracy"], ma)
    write_to_nested_dict(results, ["Neural Network", "f1-score"], f1)
    write_to_nested_dict(results, ["Neural Network", "optimals"], opts)
    print(results["Neural Network"], "\n")
    gc.collect()
    # ------------------- GAUSSIAN PROCESS --------------------
    print("\n...Gaussian Process")
    opts, ma, f1, ma_vec, f1_vec = _optimize_gaussian_process(
        X_train, y_train, X_test, y_true, cycles=cycles
    )
    write_to_nested_dict(results, ["Gaussian Process", "Accuracy"], ma)
    write_to_nested_dict(results, ["Gaussian Process", "f1-score"], f1)
    write_to_nested_dict(results, ["Gaussian Process", "optimals"], opts)
    print(results["Gaussian Process"], "\n")
    gc.collect()
    """
