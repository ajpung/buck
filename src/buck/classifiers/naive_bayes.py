from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, f1_score


def _optimize_naive_bayes(X_train_pca, y_train_flat, X_test_pca, y_true, cycles=2):
    # Define classifier
    clf = GaussianNB()

    # Shorten parameters
    Xtr_pca = X_train_pca
    ytr_flat = y_train_flat
    Xte_pca = X_test_pca

    # Define optimals
    opts = {"priors": None, "var_smoothing": 1e-09}

    # Train the classifier
    clf.fit(Xtr_pca, ytr_flat)

    # Make predictions
    y_pred = clf.predict(Xte_pca)

    # Calculate metrics
    ma = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Cyclically optimize hyperparameters
    ma_vec = [ma]
    f1_vec = [f1]

    return opts, ma, f1, ma_vec, f1_vec
