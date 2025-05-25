import gc

from buck.classifiers.ada_boost import _optimize_ada_boost
from buck.classifiers.bagging_classifier import _optimize_bagging
from buck.classifiers.decision_tree import _optimize_decision_tree
from buck.classifiers.extra_trees import _optimize_extra_trees
from buck.classifiers.gradient_boost import _optimize_gradient_boost
from buck.classifiers.k_nearest import _optimize_knn
from buck.classifiers.linear_discriminant import _optimize_linear_discriminant
from buck.classifiers.logistic_regression import _optimize_logistic_regression
from buck.classifiers.naive_bayes import _optimize_naive_bayes
from buck.classifiers.neural_network import _optimize_neural_network
from buck.classifiers.passive_aggressive import _optimize_passive_aggressive
from buck.classifiers.random_forest import _optimize_random_forest
from buck.classifiers.ridge_classifier import _optimize_ridge

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


def optimize_all(X_train_pca, y_train_flat, X_test_pca, y_true, cycles=2):
    results = {}
    print("Optimizing")
    ## -------------------- DECISION TREES ---------------------
    # print("...decision tree")
    # opts, ma, f1, ma_vec, f1_vec = _optimize_decision_tree(
    #    X_train_pca, y_train_flat, X_test_pca, y_true, cycles=cycles
    # )
    # write_to_nested_dict(results, ["Decision Tree", "Accuracy"], ma)
    # write_to_nested_dict(results, ["Decision Tree", "f1-score"], f1)
    # write_to_nested_dict(results, ["Decision Tree", "optimals"], opts)
    # print(results, "\n")
    # gc.collect()
    ## ------------------------ ADABOOST -----------------------
    # print("...adaboost")
    # opts, ma, f1, ma_vec, f1_vec = _optimize_ada_boost(
    #    X_train_pca, y_train_flat, X_test_pca, y_true, cycles=cycles
    # )
    # write_to_nested_dict(results, ["Adaboost", "Accuracy"], ma)
    # write_to_nested_dict(results, ["Adaboost", "f1-score"], f1)
    # write_to_nested_dict(results, ["Adaboost", "optimals"], opts)
    # print(results, "\n")
    # gc.collect()
    # ------------------- BAGGING CLASSIFIER ------------------
    print("...bagging classifier")
    opts, ma, f1, ma_vec, f1_vec = _optimize_bagging(
        X_train_pca, y_train_flat, X_test_pca, y_true, cycles=cycles
    )
    write_to_nested_dict(results, ["Bagging Classifier", "Accuracy"], ma)
    write_to_nested_dict(results, ["Bagging Classifier", "f1-score"], f1)
    write_to_nested_dict(results, ["Bagging Classifier", "optimals"], opts)
    print(results, "\n")
    gc.collect()
    ## ---------------------- EXTRA TREES ----------------------
    # rint("...extra trees")
    # opts, ma, f1, ma_vec, f1_vec = _optimize_extra_trees(
    #    X_train_pca, y_train_flat, X_test_pca, y_true, cycles=cycles
    # )
    # write_to_nested_dict(results, ["Extra Trees", "Accuracy"], ma)
    # write_to_nested_dict(results, ["Extra Trees", "f1-score"], f1)
    # write_to_nested_dict(results, ["Extra Trees", "optimals"], opts)
    # print(results, "\n")
    # gc.collect()
    # -------------------- GRADIENT BOOST ---------------------
    print("...gradient boost")
    opts, ma, f1, ma_vec, f1_vec = _optimize_gradient_boost(
        X_train_pca, y_train_flat, X_test_pca, y_true, cycles=cycles
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
        X_train_pca, y_train_flat, X_test_pca, y_true, cycles=cycles
    )
    write_to_nested_dict(results, ["Random Forest", "Accuracy"], ma)
    write_to_nested_dict(results, ["Random Forest", "f1-score"], f1)
    write_to_nested_dict(results, ["Random Forest", "optimals"], opts)
    print(results, "\n")
    gc.collect()
    # ------------------------ VOTING -------------------------
    print("...voting")
    ## ------------------ LINEAR DISCRIMINANT ------------------
    # print("...linear discriminant")
    # opts, ma, f1, ma_vec, f1_vec = _optimize_linear_discriminant(
    #    X_train_pca, y_train_flat, X_test_pca, y_true, cycles=cycles
    # )
    # write_to_nested_dict(results, ["Linear Discriminant", "Accuracy"], ma)
    # write_to_nested_dict(results, ["Linear Discriminant", "f1-score"], f1)
    # write_to_nested_dict(results, ["Linear Discriminant", "optimals"], opts)
    # print(results, "\n")
    # gc.collect()
    # ------------------ LOGISTIC REGRESSION ------------------
    print("...logistic regression")
    opts, ma, f1, ma_vec, f1_vec = _optimize_logistic_regression(
        X_train_pca, y_train_flat, X_test_pca, y_true, cycles=cycles
    )
    write_to_nested_dict(results, ["Logistic Regression", "Accuracy"], ma)
    write_to_nested_dict(results, ["Logistic Regression", "f1-score"], f1)
    write_to_nested_dict(results, ["Logistic Regression", "optimals"], opts)
    print(results, "\n")
    gc.collect()
    # ------------------ PASSIVE AGGRESSIVE -------------------
    print("...passive aggressive")
    opts, ma, f1, ma_vec, f1_vec = _optimize_passive_aggressive(
        X_train_pca, y_train_flat, X_test_pca, y_true, cycles=cycles
    )
    write_to_nested_dict(results, ["Passive Aggressive", "Accuracy"], ma)
    write_to_nested_dict(results, ["Passive Aggressive", "f1-score"], f1)
    write_to_nested_dict(results, ["Passive Aggressive", "optimals"], opts)
    print(results, "\n")
    gc.collect()
    # ------------------------ RIDGE --------------------------
    print("...ridge")
    opts, ma, f1, ma_vec, f1_vec = _optimize_ridge(
        X_train_pca, y_train_flat, X_test_pca, y_true, cycles=cycles
    )
    write_to_nested_dict(results, ["Ridge", "Accuracy"], ma)
    write_to_nested_dict(results, ["Ridge", "f1-score"], f1)
    write_to_nested_dict(results, ["Ridge", "optimals"], opts)
    print(results, "\n")
    gc.collect()
    # -------------- STOCHASTIC GRADIENT DESCENT --------------
    # --------------------- SELF-TRAINING ---------------------
    # ------------------------ STACKING -----------------------
    ## -------------------------- KNN --------------------------
    # print("...KNN")
    # opts, ma, f1, ma_vec, f1_vec = _optimize_knn(
    #    X_train_pca, y_train_flat, X_test_pca, y_true, cycles=cycles
    # )
    # write_to_nested_dict(results, ["KNN", "Accuracy"], ma)
    # write_to_nested_dict(results, ["KNN", "f1-score"], f1)
    # write_to_nested_dict(results, ["KNN", "optimals"], opts)
    # print(results, "\n")
    # gc.collect()
    # --------------------- NEURAL NETWORK --------------------
    print("...neural network")
    opts, ma, f1, ma_vec, f1_vec = _optimize_neural_network(
        X_train_pca, y_train_flat, X_test_pca, y_true, cycles=cycles
    )
    write_to_nested_dict(results, ["Neural Network", "Accuracy"], ma)
    write_to_nested_dict(results, ["Neural Network", "f1-score"], f1)
    write_to_nested_dict(results, ["Neural Network", "optimals"], opts)
    print(results, "\n")
    gc.collect()
    ## ---------------------- NAIVE BAYES ----------------------
    # print("...naive bayes")
    # opts, ma, f1, ma_vec, f1_vec = _optimize_naive_bayes(
    #    X_train_pca, y_train_flat, X_test_pca, y_true, cycles=cycles
    # )
    # write_to_nested_dict(results, ["Naive Bayes", "Accuracy"], ma)
    # write_to_nested_dict(results, ["Naive Bayes", "f1-score"], f1)
    # write_to_nested_dict(results, ["Naive Bayes", "optimals"], opts)
    # print(results, "\n")
    # gc.collect()
    # ------------------- GAUSSIAN PROCESS --------------------
