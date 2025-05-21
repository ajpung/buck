import gc

from buck.classifiers.ada_boost import _optimize_ada_boost
from buck.classifiers.decision_tree import _optimize_decision_tree
from buck.classifiers.extra_trees import _optimize_extra_trees
from buck.classifiers.gradient_boost import _optimize_gradient_boost
from buck.classifiers.k_nearest import _optimize_knn
from buck.classifiers.linear_discriminant import _optimize_linear_discriminant
from buck.classifiers.logistic_regression import _optimize_logistic_regression
from buck.classifiers.naive_bayes import _optimize_naive_bayes
from buck.classifiers.neural_network import _optimize_neural_network
from buck.classifiers.random_forest import _optimize_random_forest

"""
To validate:
- Neural Network
"""


def garbage_collect():
    # Force garbage collection
    gc.collect()


def optimize_all(X_train_pca, y_train_flat, X_test_pca, y_true, cycles=2):
    results = {}
    print("Optimizing")
    # ------------------------ ADABOOST -----------------------
    print("...adaboost")
    opts, ma, f1, ma_vec, f1_vec = _optimize_ada_boost(
        X_train_pca, y_train_flat, X_test_pca, y_true, cycles=cycles
    )
    results["Adaboost"]["Accuracy"] = ma
    results["Adaboost"]["f1-score"] = ma
    results["Adaboost"]["optimals"] = opts
    print(results, "\n")
    gc.collect()
    # -------------------- DECISION TREES ---------------------
    print("...decision tree")
    opts, ma, f1, ma_vec, f1_vec = _optimize_decision_tree(
        X_train_pca, y_train_flat, X_test_pca, y_true, cycles=cycles
    )
    results["Decision Tree"]["Accuracy"] = ma
    results["Decision Tree"]["f1-score"] = f1
    results["Decision Tree"]["optimals"] = opts
    print(results, "\n")
    gc.collect()
    # ---------------------- EXTRA TREES ----------------------
    print("...extra trees")
    opts, ma, f1, ma_vec, f1_vec = _optimize_extra_trees(
        X_train_pca, y_train_flat, X_test_pca, y_true, cycles=cycles
    )
    results["Extra Trees"]["Accuracy"] = ma
    results["Extra Trees"]["f1-score"] = f1
    results["Extra Trees"]["optimals"] = opts
    print(results, "\n")
    gc.collect()
    # -------------------------- KNN --------------------------
    print("...KNN")
    opts, ma, f1, ma_vec, f1_vec = _optimize_knn(
        X_train_pca, y_train_flat, X_test_pca, y_true, cycles=cycles
    )
    results["KNN"]["Accuracy"] = ma
    results["KNN"]["f1-score"] = f1
    results["KNN"]["optimals"] = opts
    print(results, "\n")
    gc.collect()
    # ------------------ LINEAR DISCRIMINANT ------------------
    print("...linear discriminant")
    opts, ma, f1, ma_vec, f1_vec = _optimize_linear_discriminant(
        X_train_pca, y_train_flat, X_test_pca, y_true, cycles=cycles
    )
    results["Linear Discriminant"]["Accuracy"] = ma
    results["Linear Discriminant"]["f1-score"] = f1
    results["Linear Discriminant"]["optimals"] = opts
    print(results, "\n")
    gc.collect()
    # ---------------------- NAIVE BAYES ----------------------
    print("...naive bayes")
    opts, ma, f1, ma_vec, f1_vec = _optimize_naive_bayes(
        X_train_pca, y_train_flat, X_test_pca, y_true, cycles=cycles
    )
    results["Naive Bayes"]["Accuracy"] = ma
    results["Naive Bayes"]["f1-score"] = f1
    results["Naive Bayes"]["optimals"] = opts
    print(results, "\n")
    gc.collect()
    # --------------------- NEURAL NETWORK --------------------
    print("...neural network")
    opts, ma, f1, ma_vec, f1_vec = _optimize_neural_network(
        X_train_pca, y_train_flat, X_test_pca, y_true, cycles=cycles
    )
    results["Neural Network"]["Accuracy"] = ma
    results["Neural Network"]["f1-score"] = f1
    results["Neural Network"]["optimals"] = opts
    print(results, "\n")
    gc.collect()
    # --------------------- RANDOM FOREST ---------------------
    print("...random forest")
    print(results, "\n")
    opts, ma, f1, ma_vec, f1_vec = _optimize_random_forest(
        X_train_pca, y_train_flat, X_test_pca, y_true, cycles=cycles
    )
    results["Random Forest"]["Accuracy"] = ma
    results["Random Forest"]["f1-score"] = f1
    results["Random Forest"]["optimals"] = opts
    print(results, "\n")
    gc.collect()
    # -------------------- GRADIENT BOOST ---------------------
    print("...gradient boost")
    opts, ma, f1, ma_vec, f1_vec = _optimize_gradient_boost(
        X_train_pca, y_train_flat, X_test_pca, y_true, cycles=cycles
    )
    results["Gradient Boost"]["Accuracy"] = ma
    results["Gradient Boost"]["f1-score"] = f1
    results["Gradient Boost"]["optimals"] = opts
    print(results, "\n")
    gc.collect()
    return results
    # ------------------ LOGISTIC REGRESSION ------------------
    print("...logistic regression")
    opts, ma, f1, ma_vec, f1_vec = _optimize_logistic_regression(
        X_train_pca, y_train_flat, X_test_pca, y_true, cycles=cycles
    )
    results["Logistic Regression"]["Accuracy"] = ma
    results["Logistic Regression"]["f1-score"] = f1
    results["Logistic Regression"]["optimals"] = opts
    print(results, "\n")
    gc.collect()
    # ------------------- BAGGING CLASSIFIER ------------------
    print("...bagging classifier")
    opts, ma, f1, ma_vec, f1_vec = _optimize_bagging_classifier(
        X_train_pca, y_train_flat, X_test_pca, y_true, cycles=cycles
    )
    results["Bagging Classifier"]["Accuracy"] = ma
    results["Bagging Classifier"]["f1-score"] = f1
    results["Bagging Classifier"]["optimals"] = opts
    print(results, "\n")
    gc.collect()
