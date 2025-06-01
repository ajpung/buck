import gc

from buck.classifiers.ada_boost import _optimize_adaboost
from buck.classifiers.bagging_classifier import _optimize_bagging
from buck.classifiers.decision_tree import _optimize_decision_tree
from buck.classifiers.extra_trees import _optimize_extra_trees
from buck.classifiers.gaussian_process import _optimize_gaussian_process
from buck.classifiers.k_nearest import _optimize_knn
from buck.classifiers.linear_discriminant import _optimize_linear_discriminant
from buck.classifiers.logistic_regression import _optimize_logistic_regression
from buck.classifiers.naive_bayes import _optimize_naive_bayes
from buck.classifiers.neural_network import _optimize_neural_network
from buck.classifiers.passive_aggressive import _optimize_passive_aggressive
from buck.classifiers.random_forest import _optimize_random_forest
from buck.classifiers.ridge_classifier import _optimize_ridge
from buck.classifiers.self_training import _optimize_selftrain
from buck.classifiers.stacking_classifier import _optimize_stacking
from buck.classifiers.stocastic_gradient_descent import _optimize_sgd
from buck.classifiers.voting_classifier import _optimize_voting
from buck.classifiers.convolution_nn import _optimize_cnn
from buck.classifiers.xg_boost import _optimize_xgboost
from buck.classifiers.cat_boost import _optimize_catboost
from buck.classifiers.light_gbm import _optimize_lightgbm

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
    opts, ma, f1, ma_vec, f1_vec = _optimize_ada_boost(
        X_train, y_train, X_test, y_true, cycles=cycles
    )
    write_to_nested_dict(results, ["Adaboost", "Accuracy"], ma)
    write_to_nested_dict(results, ["Adaboost", "f1-score"], f1)
    write_to_nested_dict(results, ["Adaboost", "optimals"], opts)
    print(results["Adaboost"], "\n")
    gc.collect()
    # ------------------- BAGGING CLASSIFIER ------------------
    opts, ma, f1, ma_vec, f1_vec = _optimize_bagging(
        X_train, y_train, X_test, y_true, cycles=cycles
    )
    write_to_nested_dict(results, ["Bagging Classifier", "Accuracy"], ma)
    write_to_nested_dict(results, ["Bagging Classifier", "f1-score"], f1)
    write_to_nested_dict(results, ["Bagging Classifier", "optimals"], opts)
    print(results["Bagging Classifier"], "\n")
    gc.collect()
    # -------------------- DECISION TREES ---------------------
    opts, ma, f1, ma_vec, f1_vec = _optimize_decision_tree(
        X_train, y_train, X_test, y_true, cycles=cycles
    )
    write_to_nested_dict(results, ["Decision Tree", "Accuracy"], ma)
    write_to_nested_dict(results, ["Decision Tree", "f1-score"], f1)
    write_to_nested_dict(results, ["Decision Tree", "optimals"], opts)
    print(results["Decision Tree"], "\n")
    gc.collect()
    # ---------------------- EXTRA TREES ----------------------
    opts, ma, f1, ma_vec, f1_vec = _optimize_extra_trees(
        X_train, y_train, X_test, y_true, cycles=cycles
    )
    write_to_nested_dict(results, ["Extra Trees", "Accuracy"], ma)
    write_to_nested_dict(results, ["Extra Trees", "f1-score"], f1)
    write_to_nested_dict(results, ["Extra Trees", "optimals"], opts)
    print(results["Extra Trees"], "\n")
    gc.collect()
    # ----------------------- XG BOOST ------------------------
    opts, ma, f1, ma_vec, f1_vec = _optimize_xgboost(
        X_train, y_train, X_test, y_true, cycles=cycles
    )
    write_to_nested_dict(results, ["XG Boost", "Accuracy"], ma)
    write_to_nested_dict(results, ["XG Boost", "f1-score"], f1)
    write_to_nested_dict(results, ["XG Boost", "optimals"], opts)
    print(results["XG Boost"], "\n")
    gc.collect()
    # ----------------------- CAT BOOST -----------------------
    opts, ma, f1, ma_vec, f1_vec = _optimize_catboost(
        X_train, y_train, X_test, y_true, cycles=cycles
    )
    write_to_nested_dict(results, ["Cat Boost", "Accuracy"], ma)
    write_to_nested_dict(results, ["Cat Boost", "f1-score"], f1)
    write_to_nested_dict(results, ["Cat Boost", "optimals"], opts)
    print(results["Cat Boost"], "\n")
    gc.collect()
    return results
    # ----------------------- LIGHT GBM -----------------------
    opts, ma, f1, ma_vec, f1_vec = _optimize_lightgbm(
        X_train, y_train, X_test, y_true, cycles=cycles
    )
    write_to_nested_dict(results, ["Light GBM", "Accuracy"], ma)
    write_to_nested_dict(results, ["Light GBM", "f1-score"], f1)
    write_to_nested_dict(results, ["Light GBM", "optimals"], opts)
    print(results["Light GBM"], "\n")
    gc.collect()
    # --------------------- RANDOM FOREST ---------------------
    opts, ma, f1, ma_vec, f1_vec = _optimize_random_forest(
        X_train, y_train, X_test, y_true, cycles=cycles
    )
    write_to_nested_dict(results, ["Random Forest", "Accuracy"], ma)
    write_to_nested_dict(results, ["Random Forest", "f1-score"], f1)
    write_to_nested_dict(results, ["Random Forest", "optimals"], opts)
    print(results["Random Forest"], "\n")
    gc.collect()
    # ------------------ LINEAR DISCRIMINANT ------------------
    opts, ma, f1, ma_vec, f1_vec = _optimize_linear_discriminant(
        X_train, y_train, X_test, y_true, cycles=cycles
    )
    write_to_nested_dict(results, ["Linear Discriminant", "Accuracy"], ma)
    write_to_nested_dict(results, ["Linear Discriminant", "f1-score"], f1)
    write_to_nested_dict(results, ["Linear Discriminant", "optimals"], opts)
    print(results["Linear Discriminant"], "\n")
    gc.collect()
    # ------------------ LOGISTIC REGRESSION ------------------
    opts, ma, f1, ma_vec, f1_vec = _optimize_logistic_regression(
        X_train, y_train, X_test, y_true, cycles=cycles
    )
    write_to_nested_dict(results, ["Logistic Regression", "Accuracy"], ma)
    write_to_nested_dict(results, ["Logistic Regression", "f1-score"], f1)
    write_to_nested_dict(results, ["Logistic Regression", "optimals"], opts)
    print(results["Logistic Regression"], "\n")
    gc.collect()
    # ------------------ PASSIVE AGGRESSIVE -------------------
    opts, ma, f1, ma_vec, f1_vec = _optimize_passive_aggressive(
        X_train, y_train, X_test, y_true, cycles=cycles
    )
    write_to_nested_dict(results, ["Passive Aggressive", "Accuracy"], ma)
    write_to_nested_dict(results, ["Passive Aggressive", "f1-score"], f1)
    write_to_nested_dict(results, ["Passive Aggressive", "optimals"], opts)
    print(results["Passive Aggressive"], "\n")
    gc.collect()
    # ------------------------ RIDGE --------------------------
    opts, ma, f1, ma_vec, f1_vec = _optimize_ridge(
        X_train, y_train, X_test, y_true, cycles=cycles
    )
    write_to_nested_dict(results, ["Ridge", "Accuracy"], ma)
    write_to_nested_dict(results, ["Ridge", "f1-score"], f1)
    write_to_nested_dict(results, ["Ridge", "optimals"], opts)
    print(results["Ridge"], "\n")
    gc.collect()
    # --------------------- SELF-TRAINING ---------------------
    opts, ma, f1, ma_vec, f1_vec = _optimize_selftrain(
        X_train, y_train, X_test, y_true, cycles=cycles
    )
    write_to_nested_dict(results, ["Naive Bayes", "Accuracy"], ma)
    write_to_nested_dict(results, ["Naive Bayes", "f1-score"], f1)
    write_to_nested_dict(results, ["Naive Bayes", "optimals"], opts)
    print(results["Naive Bayes"], "\n")
    gc.collect()
    # -------------- STOCHASTIC GRADIENT DESCENT --------------
    opts, ma, f1, ma_vec, f1_vec = _optimize_sgd(
        X_train, y_train, X_test, y_true, cycles=cycles
    )
    write_to_nested_dict(results, ["Stochastic Gradient Descent", "Accuracy"], ma)
    write_to_nested_dict(results, ["Stochastic Gradient Descent", "f1-score"], f1)
    write_to_nested_dict(results, ["Stochastic Gradient Descent", "optimals"], opts)
    print(results["Stochastic Gradient Descent"], "\n")
    gc.collect()
    # ------------------------ STACKING -----------------------
    opts, ma, f1, ma_vec, f1_vec = _optimize_stacking(
        X_train, y_train, X_test, y_true, cycles=cycles
    )
    write_to_nested_dict(results, ["Stacking Classifier", "Accuracy"], ma)
    write_to_nested_dict(results, ["Stacking Classifier", "f1-score"], f1)
    write_to_nested_dict(results, ["Stacking Classifier", "optimals"], opts)
    print(results["Stacking Classifier"], "\n")
    gc.collect()
    # ---------------------- NAIVE BAYES ----------------------
    opts, ma, f1, ma_vec, f1_vec = _optimize_naive_bayes(
        X_train, y_train, X_test, y_true, cycles=cycles
    )
    write_to_nested_dict(results, ["Naive Bayes", "Accuracy"], ma)
    write_to_nested_dict(results, ["Naive Bayes", "f1-score"], f1)
    write_to_nested_dict(results, ["Naive Bayes", "optimals"], opts)
    print(results["Naive Bayes"], "\n")
    gc.collect()
    # -------------------------- KNN --------------------------
    opts, ma, f1, ma_vec, f1_vec = _optimize_knn(
        X_train, y_train, X_test, y_true, cycles=cycles
    )
    write_to_nested_dict(results, ["KNN", "Accuracy"], ma)
    write_to_nested_dict(results, ["KNN", "f1-score"], f1)
    write_to_nested_dict(results, ["KNN", "optimals"], opts)
    print(results["KNN"], "\n")
    gc.collect()
    # --------------------- NEURAL NETWORK --------------------
    opts, ma, f1, ma_vec, f1_vec = _optimize_neural_network(
        X_train, y_train, X_test, y_true, cycles=cycles
    )
    write_to_nested_dict(results, ["Neural Network", "Accuracy"], ma)
    write_to_nested_dict(results, ["Neural Network", "f1-score"], f1)
    write_to_nested_dict(results, ["Neural Network", "optimals"], opts)
    print(results["Neural Network"], "\n")
    gc.collect()
    # --------------- CONVOLUTION NEURAL NETWORK --------------
    opts, ma, f1, ma_vec, f1_vec = _optimize_cnn(
        X_train, y_train, X_test, y_true, cycles=cycles
    )
    write_to_nested_dict(results, ["Convolution Neural Network", "Accuracy"], ma)
    write_to_nested_dict(results, ["Convolution Neural Network", "f1-score"], f1)
    write_to_nested_dict(results, ["Convolution Neural Network", "optimals"], opts)
    print(results["Convolution Neural Network"], "\n")
    gc.collect()
    # ------------------- GAUSSIAN PROCESS --------------------
    opts, ma, f1, ma_vec, f1_vec = _optimize_gaussian_process(
        X_train, y_train, X_test, y_true, cycles=cycles
    )
    write_to_nested_dict(results, ["Gaussian Process", "Accuracy"], ma)
    write_to_nested_dict(results, ["Gaussian Process", "f1-score"], f1)
    write_to_nested_dict(results, ["Gaussian Process", "optimals"], opts)
    print(results["Gaussian Process"], "\n")
    gc.collect()
    # ------------------------ VOTING -------------------------
    opts, ma, f1, ma_vec, f1_vec = _optimize_voting(
        X_train, y_train, X_test, y_true, cycles=cycles
    )
    write_to_nested_dict(results, ["Voting Classifier", "Accuracy"], ma)
    write_to_nested_dict(results, ["Voting Classifier", "f1-score"], f1)
    write_to_nested_dict(results, ["Voting Classifier", "optimals"], opts)
    print(results["Voting Classifier"], "\n")
    gc.collect()
