import os

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    BaggingClassifier,
    StackingClassifier,
    VotingClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import (
    LogisticRegression,
    PassiveAggressiveClassifier,
    RidgeClassifier,
    SGDClassifier,
)
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.semi_supervised import SelfTrainingClassifier


def compare_models(
    X_train_pca, y_train_flat, X_test_pca, y_true, num_classes, label_mapping, cores=6
):
    # Define number of cores to use
    os.environ["LOKY_MAX_CPU_COUNT"] = str(cores)

    # Define classifiers to test
    classifiers = {
        "Passive Aggressive": PassiveAggressiveClassifier(),
        "Ridge Classifier": RidgeClassifier(),
        "SGD Classifier": SGDClassifier(),
        "Self Training": SelfTrainingClassifier(),
        "Bagging": BaggingClassifier(random_state=42),
        "Stacking": StackingClassifier(random_state=42),
        "Voting": VotingClassifier(random_state=42),
        "Gaussian Process": GaussianProcessClassifier(),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=min(8, len(X_train_pca)), weights="distance"
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=5, class_weight="balanced", random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=3, random_state=42
        ),
        "Logistic Regression": LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=10000, random_state=42
        ),
        "Neural Network": MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation="relu",
            solver="adam",
            alpha=0.001,
            random_state=42,
            max_iter=1000,
        ),
        "Naive Bayes": GaussianNB(),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=100, max_depth=5, class_weight="balanced", random_state=42
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=100, learning_rate=1.0, random_state=42
        ),
        "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
    }

    # Store results
    results = {}
    all_predictions = {}
    all_f1_scores = {}
    classes_predicted = {}
    classification_reports = {}

    print("\nEvaluating classifiers...")

    # Evaluate each classifier
    for name, clf in classifiers.items():
        # Train the classifier
        clf.fit(X_train_pca, y_train_flat)

        # Make predictions
        y_pred = clf.predict(X_test_pca)

        # Store predictions
        all_predictions[name] = y_pred

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        unique_preds = len(np.unique(y_pred))

        # Store results
        results[name] = accuracy
        all_f1_scores[name] = f1
        classes_predicted[name] = unique_preds
        reverse_mapping = {i: label for label, i in label_mapping.items()}

        # Generate classification report
        class_report = classification_report(
            y_true,
            y_pred,
            target_names=[f"Age {reverse_mapping[i]}" for i in range(num_classes)],
            zero_division=0,
        )
        classification_reports[name] = class_report

    # Create comparison table
    comparison_df = pd.DataFrame(
        {
            "Classifier": list(classifiers.keys()),
            "Accuracy": [results[name] for name in classifiers],
            "F1 Score": [all_f1_scores[name] for name in classifiers],
            "Classes Predicted": [classes_predicted[name] for name in classifiers],
            "Predicted Ages": [
                ", ".join(
                    [
                        str(reverse_mapping[p])
                        for p in sorted(set(all_predictions[name]))
                    ]
                )
                for name in classifiers
            ],
        }
    )

    # Sort by number of classes predicted first, then by accuracy
    comparison_df = comparison_df.sort_values(
        ["Classes Predicted", "Accuracy"], ascending=[False, False]
    ).reset_index(drop=True)

    # Display comparison table
    print("=" * 50)
    print("\nCLASSIFIER COMPARISON")
    print("=" * 50)
    print(comparison_df.to_string(index=False))

    # Find best classifier (prioritizing diversity, then accuracy)
    best_classifier = comparison_df.iloc[0]
    print(f"\n" + "=" * 50)
    print("BEST CLASSIFIER")
    print("=" * 50)
    print(f"Classifier: {best_classifier['Classifier']}")
    print(f"Accuracy: {best_classifier['Accuracy']:.4f}")
    print(f"F1 Score: {best_classifier['F1 Score']:.4f}")
    print(f"Classes Predicted: {best_classifier['Classes Predicted']}")
    print(f"Predicted Ages: {best_classifier['Predicted Ages']}")

    # Show detailed classification report for best classifier
    print(f"\nDetailed Classification Report for {best_classifier['Classifier']}:")
    print(classification_reports[best_classifier["Classifier"]])
    return best_classifier["Classifier"]
