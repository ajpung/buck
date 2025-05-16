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


def optimize_all(X_train_pca, y_train_flat, X_test_pca, y_true, cycles=5):
    results = {}
    print("Optimizing")
    print("...adaboost")
    optimals, accuracy = _optimize_ada_boost(
        X_train_pca, y_train_flat, X_test_pca, y_true, cycles=cycles
    )
    results["Adaboost"] = accuracy
    gc.collect()
    print("...decision tree")
    optimals, accuracy = _optimize_decision_tree(
        X_train_pca, y_train_flat, X_test_pca, y_true, cycles=cycles
    )
    results["Decision Tree"] = accuracy
    gc.collect()
    print("...extra trees")
    optimals, accuracy = _optimize_extra_trees(
        X_train_pca, y_train_flat, X_test_pca, y_true, cycles=cycles
    )
    results["Extra Trees"] = accuracy
    gc.collect()
    print("...gradient boost")
    optimals, accuracy = _optimize_gradient_boost(
        X_train_pca, y_train_flat, X_test_pca, y_true, cycles=cycles
    )
    results["Gradient Boosting"] = accuracy
    gc.collect()
    print("...KNN")
    optimals, accuracy = _optimize_knn(
        X_train_pca, y_train_flat, X_test_pca, y_true, cycles=cycles
    )
    results["KNN"] = accuracy
    gc.collect()
    print("...linear discriminant")
    optimals, accuracy = _optimize_linear_discriminant(
        X_train_pca, y_train_flat, X_test_pca, y_true, cycles=cycles
    )
    results["Linear Discriminant"] = accuracy
    gc.collect()
    print("...logistic regression")
    optimals, accuracy = _optimize_logistic_regression(
        X_train_pca, y_train_flat, X_test_pca, y_true, cycles=cycles
    )
    results["Logistic Regression"] = accuracy
    gc.collect()
    print("...naive bayes")
    optimals, accuracy = _optimize_naive_bayes(
        X_train_pca, y_train_flat, X_test_pca, y_true, cycles=cycles
    )
    results["Naive Bayes"] = accuracy
    gc.collect()
    print("...neural network")
    optimals, accuracy = _optimize_neural_network(
        X_train_pca, y_train_flat, X_test_pca, y_true, cycles=cycles
    )
    results["Neural Network"] = accuracy
    gc.collect()
    print("...random forest")
    optimals, accuracy = _optimize_random_forest(
        X_train_pca, y_train_flat, X_test_pca, y_true, cycles=cycles
    )
    results["Random Forest"] = accuracy
    gc.collect()
    return results
