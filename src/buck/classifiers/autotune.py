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
- *Gradient Boosting
- *Linear Discriminant Analysis
- *Logistic Regression
- *Naive Bayes
- Neural Network
"""


def agnostic_optimizer(
    X_train_pca,
    y_train_flat,
    X_test_pca,
    y_true,
    best: str,
):
    """
    Optimizes the classifier based on the best classifier selected.
    :param X_train_pca: PCA transformed training data
    :param y_train_flat: Flattened training labels
    :param X_test_pca: PCA transformed test data
    :param y_true: True labels for the test data
    :param best: The best classifier selected
    :return: None
    """
    if best == "AdaBoost":
        optimals, accuracy = _optimize_ada_boost(
            X_train_pca, y_train_flat, X_test_pca, y_true
        )
    elif best == "Decision Tree":
        optimals, accuracy = _optimize_decision_tree(
            X_train_pca, y_train_flat, X_test_pca, y_true
        )
    elif best == "Extra Trees":
        optimals, accuracy = _optimize_extra_trees(
            X_train_pca, y_train_flat, X_test_pca, y_true
        )
    elif best == "Gradient Boosting":
        optimals, accuracy = _optimize_gradient_boost(
            X_train_pca, y_train_flat, X_test_pca, y_true
        )
    elif best == "K-Nearest Neighbors":
        optimals, accuracy = _optimize_knn(
            X_train_pca, y_train_flat, X_test_pca, y_true
        )
    elif best == "Linear Discriminant Analysis":
        optimals, accuracy = _optimize_linear_discriminant(
            X_train_pca, y_train_flat, X_test_pca, y_true
        )
    elif best == "Logistic Regression":
        optimals, accuracy = _optimize_logistic_regression(
            X_train_pca, y_train_flat, X_test_pca, y_true
        )
    elif best == "Naive Bayes":
        optimals, accuracy = _optimize_naive_bayes(
            X_train_pca, y_train_flat, X_test_pca, y_true
        )
    elif best == "Neural Network":
        optimals, accuracy = _optimize_neural_network(
            X_train_pca, y_train_flat, X_test_pca, y_true
        )
    elif best == "Random Forest":
        optimals, accuracy = _optimize_random_forest(
            X_train_pca, y_train_flat, X_test_pca, y_true
        )

    return best, optimals, accuracy
