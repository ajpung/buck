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
    """
    # ------------------------ ADABOOST -----------------------
    print("...adaboost")
    optimals, accuracy = _optimize_ada_boost(
        X_train_pca, y_train_flat, X_test_pca, y_true, cycles=cycles
    )
    results["Adaboost"] = accuracy
    print(optimals)
    print(results,'\n')
    gc.collect()
    # -------------------- DECISION TREES ---------------------
    print("...decision tree")
    optimals, accuracy = _optimize_decision_tree(
        X_train_pca, y_train_flat, X_test_pca, y_true, cycles=cycles
    )
    results["Decision Tree"] = accuracy
    print(optimals)
    print(results,'\n')
    gc.collect()
    # ---------------------- EXTRA TREES ----------------------
    print("...extra trees")
    optimals, accuracy = _optimize_extra_trees(
        X_train_pca, y_train_flat, X_test_pca, y_true, cycles=cycles
    )
    results["Extra Trees"] = accuracy
    print(optimals)
    print(results,'\n')
    gc.collect()

    # -------------------------- KNN --------------------------
    print("...KNN")
    optimals, accuracy = _optimize_knn(
        X_train_pca, y_train_flat, X_test_pca, y_true, cycles=cycles
    )
    results["KNN"] = accuracy
    print(optimals)
    print(results,'\n')
    gc.collect()

    # ------------------ LINEAR DISCRIMINANT ------------------
    print("...linear discriminant")
    optimals, accuracy = _optimize_linear_discriminant(
        X_train_pca, y_train_flat, X_test_pca, y_true, cycles=cycles
    )
    results["Linear Discriminant"] = accuracy
    print(optimals)
    print(results,'\n')
    gc.collect()

    # ---------------------- NAIVE BAYES ----------------------
    print("...naive bayes")
    optimals, accuracy = _optimize_naive_bayes(
        X_train_pca, y_train_flat, X_test_pca, y_true, cycles=cycles
    )
    results["Naive Bayes"] = accuracy
    print(optimals)
    print(results, "\n")
    gc.collect()
    """

    # --------------------- NEURAL NETWORK --------------------
    print("...neural network")
    optimals, accuracy = _optimize_neural_network(
        X_train_pca, y_train_flat, X_test_pca, y_true, cycles=cycles
    )
    results["Neural Network"] = accuracy
    print(optimals)
    print(results, "\n")
    gc.collect()
    # --------------------- RANDOM FOREST ---------------------
    print("...random forest")
    print(results, "\n")
    optimals, accuracy = _optimize_random_forest(
        X_train_pca, y_train_flat, X_test_pca, y_true, cycles=cycles
    )
    results["Random Forest"] = accuracy
    print(optimals)
    print(results, "\n")
    gc.collect()
    # -------------------- GRADIENT BOOST ---------------------
    print("...gradient boost")
    optimals, accuracy = _optimize_gradient_boost(
        X_train_pca, y_train_flat, X_test_pca, y_true, cycles=cycles
    )
    results["Gradient Boosting"] = accuracy
    print(optimals)
    print(results, "\n")
    gc.collect()
    return results
    # ------------------ LOGISTIC REGRESSION ------------------
    print("...logistic regression")
    optimals, accuracy = _optimize_logistic_regression(
        X_train_pca, y_train_flat, X_test_pca, y_true, cycles=cycles
    )
    results["Logistic Regression"] = accuracy
    print(optimals)
    print(results, "\n")
    gc.collect()
