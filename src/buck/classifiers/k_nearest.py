from typing import Any
import warnings
import numpy as np
import time
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.exceptions import ConvergenceWarning

# Suppress warnings for cleaner progress bars
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Global efficiency controls
_max_time_per_step = 300  # 5 minutes max per optimization step
_max_time_per_model = (
    60  # 1 minute max per model evaluation (KNN can be slow on large data)
)
_min_accuracy_threshold = 0.15  # Stop if accuracy is consistently terrible
_consecutive_failures = 0
_max_consecutive_failures = 3


def _safe_evaluate_model(X_train, y_train, X_test, y_true, **kwargs):
    """Safely evaluate a KNN model configuration with timeout protection"""
    global _consecutive_failures

    try:
        start_time = time.time()
        classifier = KNeighborsClassifier(**kwargs)
        classifier.fit(X_train, y_train)  # Very fast for KNN

        # Check if prediction is taking too long (the slow part for KNN)
        pred_start = time.time()
        y_pred = classifier.predict(X_test)
        pred_time = time.time() - pred_start

        total_time = time.time() - start_time
        if total_time > _max_time_per_model:
            print(
                f"⏰ KNN timeout after {total_time:.1f}s (prediction: {pred_time:.1f}s)"
            )
            return 0.0, 0.0, False

        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        # Track consecutive failures
        if accuracy < _min_accuracy_threshold:
            _consecutive_failures += 1
        else:
            _consecutive_failures = 0

        return accuracy, f1, True
    except Exception as e:
        _consecutive_failures += 1
        return 0.0, 0.0, False


def _get_smart_k_range(n_samples, n_features):
    """Get smart k range based on dataset characteristics"""

    # Rule of thumb: k should be roughly sqrt(n_samples)
    optimal_k_approx = int(np.sqrt(n_samples))

    if n_samples < 100:
        # Very small dataset
        k_range = list(range(1, min(21, n_samples // 2)))
    elif n_samples < 1000:
        # Small dataset - test around optimal
        k_range = list(range(1, 11)) + list(range(15, min(51, n_samples // 4), 5))
    elif n_samples < 10000:
        # Medium dataset - focused around optimal
        start_k = max(1, optimal_k_approx - 10)
        end_k = min(optimal_k_approx + 20, n_samples // 10)
        k_range = list(range(1, 11)) + list(range(start_k, end_k, 3))
    else:
        # Large dataset - very focused range (KNN gets slow)
        start_k = max(1, optimal_k_approx - 5)
        end_k = min(optimal_k_approx + 10, 50)  # Cap at 50 for speed
        k_range = list(range(1, 8)) + list(range(start_k, end_k, 2))

    # Remove duplicates and sort
    k_range = sorted(list(set(k_range)))

    return k_range


def _get_smart_metrics(n_samples, n_features):
    """Get smart distance metrics based on dataset characteristics"""

    # Fast and generally effective metrics first
    fast_metrics = ["euclidean", "manhattan", "minkowski"]

    # Add more metrics for smaller datasets
    if n_samples < 1000:
        additional_metrics = ["chebyshev", "cosine", "correlation"]
    elif n_samples < 5000:
        additional_metrics = ["chebyshev", "cosine"]
    else:
        additional_metrics = []  # Only fast metrics for large datasets

    # Special metrics for binary/sparse data
    if n_features > 100 or np.all((X_train >= 0) & (X_train <= 1)):
        if n_samples < 2000:  # Only for smaller datasets due to speed
            additional_metrics.extend(["hamming", "jaccard"])

    return fast_metrics + additional_metrics


def _optimize_nn(X_train, y_train, X_test, y_true, opts):
    """Optimize number of neighbors with smart range selection"""
    global _consecutive_failures
    start_time = time.time()
    _consecutive_failures = 0

    max_acc = -np.inf
    best_f1 = 0.0
    n_samples, n_features = X_train.shape

    # Get smart k range based on dataset size
    variable_array = _get_smart_k_range(n_samples, n_features)
    best_val = 5  # Good default

    with tqdm(
        variable_array, desc="Optimizing Number of Neighbors", leave=False
    ) as pbar:
        for v in pbar:
            # Early stopping conditions
            if time.time() - start_time > _max_time_per_step:
                pbar.set_description("Number of Neighbors (TIME LIMIT)")
                break
            if _consecutive_failures >= _max_consecutive_failures:
                pbar.set_description("Number of Neighbors (POOR ACCURACY)")
                break

            test_opts = opts.copy()
            test_opts["n_neighbors"] = v

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, **test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_val = v

            pbar.set_postfix(
                {
                    "k": v,
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts["n_neighbors"] = best_val
    return opts, max_acc, best_f1


def _optimize_weights_and_algorithm(X_train, y_train, X_test, y_true, opts):
    """Optimize weights and algorithm together for efficiency"""
    global _consecutive_failures
    start_time = time.time()
    _consecutive_failures = 0

    max_acc = -np.inf
    best_f1 = 0.0
    n_samples = X_train.shape[0]

    # Combined weight-algorithm configurations
    if n_samples < 1000:
        configs = [
            {"weights": "uniform", "algorithm": "auto"},
            {"weights": "distance", "algorithm": "auto"},
            {"weights": "uniform", "algorithm": "ball_tree"},
            {"weights": "distance", "algorithm": "ball_tree"},
            {"weights": "uniform", "algorithm": "kd_tree"},
            {"weights": "distance", "algorithm": "kd_tree"},
        ]
    elif n_samples < 10000:
        configs = [
            {"weights": "uniform", "algorithm": "auto"},
            {"weights": "distance", "algorithm": "auto"},
            {"weights": "uniform", "algorithm": "ball_tree"},
            {"weights": "distance", "algorithm": "ball_tree"},
        ]
    else:
        # Large datasets - focus on fastest options
        configs = [
            {"weights": "uniform", "algorithm": "auto"},
            {"weights": "distance", "algorithm": "auto"},
        ]

    best_config = configs[0]

    with tqdm(configs, desc="Optimizing Weights & Algorithm", leave=False) as pbar:
        for config in pbar:
            # Early stopping conditions
            if time.time() - start_time > _max_time_per_step:
                pbar.set_description("Weights & Algorithm (TIME LIMIT)")
                break
            if _consecutive_failures >= _max_consecutive_failures:
                pbar.set_description("Weights & Algorithm (POOR ACCURACY)")
                break

            test_opts = opts.copy()
            test_opts.update(config)

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, **test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_config = config

            pbar.set_postfix(
                {
                    "weights": config["weights"][:4],
                    "algo": config["algorithm"][:4],
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts.update(best_config)
    return opts, max_acc, best_f1


def _optimize_leaf_size(X_train, y_train, X_test, y_true, opts):
    """Optimize leaf size with smart range"""
    global _consecutive_failures
    start_time = time.time()
    _consecutive_failures = 0

    max_acc = -np.inf
    best_f1 = 0.0
    n_samples = X_train.shape[0]

    # Smart leaf size range based on dataset size
    if n_samples < 1000:
        variable_array = [1, 5, 10, 20, 30, 50]
    elif n_samples < 10000:
        variable_array = [10, 20, 30, 50, 70, 100]
    else:
        variable_array = [30, 50, 70, 100, 150]  # Larger leaf sizes for big datasets

    best_val = 30  # sklearn default

    with tqdm(variable_array, desc="Optimizing Leaf Size", leave=False) as pbar:
        for v in pbar:
            # Early stopping conditions
            if time.time() - start_time > _max_time_per_step:
                pbar.set_description("Leaf Size (TIME LIMIT)")
                break
            if _consecutive_failures >= _max_consecutive_failures:
                pbar.set_description("Leaf Size (POOR ACCURACY)")
                break

            test_opts = opts.copy()
            test_opts["leaf_size"] = v

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, **test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_val = v

            pbar.set_postfix(
                {
                    "leaf_size": v,
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts["leaf_size"] = best_val
    return opts, max_acc, best_f1


def _optimize_metric_and_p(X_train, y_train, X_test, y_true, opts):
    """Optimize distance metric and p parameter together"""
    global _consecutive_failures
    start_time = time.time()
    _consecutive_failures = 0

    max_acc = -np.inf
    best_f1 = 0.0
    n_samples, n_features = X_train.shape

    # Get smart metrics based on dataset characteristics
    metrics = _get_smart_metrics(n_samples, n_features)

    # Create metric-p combinations
    configs = []
    for metric in metrics:
        if metric == "minkowski":
            # Test different p values for Minkowski
            for p in [1, 2, 3]:  # 1=Manhattan, 2=Euclidean, 3=higher order
                configs.append({"metric": metric, "p": p})
        else:
            configs.append({"metric": metric, "p": 2})  # p is ignored for non-Minkowski

    best_config = {"metric": "euclidean", "p": 2}

    with tqdm(configs, desc="Optimizing Distance Metric", leave=False) as pbar:
        for config in pbar:
            # Early stopping conditions
            if time.time() - start_time > _max_time_per_step:
                pbar.set_description("Distance Metric (TIME LIMIT)")
                break
            if _consecutive_failures >= _max_consecutive_failures:
                pbar.set_description("Distance Metric (POOR ACCURACY)")
                break

            test_opts = opts.copy()
            test_opts.update(config)

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, **test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_config = config

            metric_display = config["metric"][:8]
            if config["metric"] == "minkowski":
                metric_display += f"(p={config['p']})"

            pbar.set_postfix(
                {
                    "metric": metric_display[:12],
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts.update(best_config)
    return opts, max_acc, best_f1


def _optimize_knn(X_train, y_train, X_test, y_true, cycles=2):
    """
    FAST optimized hyperparameters for KNeighborsClassifier.
    :param X_train: Training data
    :param y_train: Training labels
    :param X_test: Test data
    :param y_true: True labels for the test data
    :param cycles: Number of optimization cycles
    """

    n_samples, n_features = X_train.shape
    print(f"Dataset: {n_samples} samples, {n_features} features")

    # KNN scalability warning
    if n_samples > 10000:
        print("⚠️  WARNING: KNN is slow on large datasets!")
        print(f"With {n_samples} samples, prediction time will be significant.")
        print(
            "Consider using faster algorithms (XGBoost, RandomForest) for large data."
        )

    if n_features > 50:
        print("⚠️  WARNING: KNN suffers from curse of dimensionality!")
        print(f"With {n_features} features, performance may be poor.")
        print("Consider dimensionality reduction or other algorithms.")

    # Quick baseline check
    print("Running baseline check...")
    baseline_knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    start_time = time.time()
    baseline_knn.fit(X_train, y_train)
    fit_time = time.time() - start_time

    start_time = time.time()
    baseline_acc = baseline_knn.score(X_test, y_true)
    pred_time = time.time() - start_time

    print(f"Baseline KNN (k=5): {baseline_acc:.4f}")
    print(f"Training time: {fit_time:.3f}s, Prediction time: {pred_time:.3f}s")

    if pred_time > 10:
        print(
            "⚠️  Very slow prediction time - consider reducing dataset size or using different algorithm"
        )

    if baseline_acc < 0.1:
        print("⚠️  WARNING: Very low baseline accuracy!")
        print(
            "KNN may not be suitable for this dataset. Consider feature scaling or different algorithms."
        )

    # Define initial parameters with good defaults
    opts = {
        "n_neighbors": 5,
        "weights": "uniform",
        "algorithm": "auto",
        "leaf_size": 30,
        "p": 2,
        "metric": "euclidean",
        "metric_params": None,
        "n_jobs": -1,  # Use all available cores
    }

    # Track results
    ma_vec = []
    f1_vec = []

    # Main optimization loop
    with tqdm(range(cycles), desc="FAST KNN Optimization", position=0) as cycle_pbar:
        for c in cycle_pbar:
            cycle_start_time = time.time()
            cycle_pbar.set_description(f"KNN Cycle {c + 1}/{cycles}")

            # Optimizations in order of importance for KNN
            opts, _, _ = _optimize_nn(X_train, y_train, X_test, y_true, opts)
            opts, _, _ = _optimize_weights_and_algorithm(
                X_train, y_train, X_test, y_true, opts
            )
            opts, _, _ = _optimize_metric_and_p(X_train, y_train, X_test, y_true, opts)
            opts, ma, f1 = _optimize_leaf_size(X_train, y_train, X_test, y_true, opts)

            ma_vec.append(ma)
            f1_vec.append(f1)

            cycle_time = time.time() - cycle_start_time

            cycle_pbar.set_postfix(
                {
                    "accuracy": f"{ma:.4f}",
                    "f1": f"{f1:.4f}",
                    "best_overall": f"{max(ma_vec):.4f}",
                    "cycle_time": f"{cycle_time:.1f}s",
                    "k": opts["n_neighbors"],
                    "metric": opts["metric"][:8],
                    "weights": opts["weights"][:4],
                    "algorithm": opts["algorithm"][:4],
                    "baseline_beat": f"{ma - baseline_acc:+.4f}",
                }
            )

    return opts, ma, f1, ma_vec, f1_vec


def _analyze_knn_performance(X_train, y_train, X_test, y_true, best_opts):
    """Analyze KNN performance and provide insights"""

    print("\n" + "=" * 60)
    print("FAST KNN PERFORMANCE ANALYSIS")
    print("=" * 60)

    # Train the optimal model and measure timing
    print("Training final KNN model with best parameters...")

    start_time = time.time()
    knn_clf = KNeighborsClassifier(**best_opts)
    knn_clf.fit(X_train, y_train)
    fit_time = time.time() - start_time

    start_time = time.time()
    y_pred = knn_clf.predict(X_test)
    pred_time = time.time() - start_time

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")

    print(f"\n🎯 Optimal KNN Performance:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Training Time: {fit_time:.3f} seconds")
    print(f"  Prediction Time: {pred_time:.3f} seconds")

    # Compare with baseline
    baseline_knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    baseline_knn.fit(X_train, y_train)
    baseline_acc = baseline_knn.score(X_test, y_true)
    baseline_f1 = f1_score(y_true, baseline_knn.predict(X_test), average="weighted")

    print(f"\n📊 Comparison with Baseline (k=5):")
    print(f"  Baseline Accuracy: {baseline_acc:.4f}")
    print(f"  Baseline F1: {baseline_f1:.4f}")
    print(
        f"  Improvement: {accuracy - baseline_acc:+.4f} accuracy, {f1 - baseline_f1:+.4f} F1"
    )

    # Configuration analysis
    print(f"\n🔧 Optimal Configuration:")
    print(f"  Number of Neighbors (k): {best_opts['n_neighbors']}")
    print(f"  Distance Metric: {best_opts['metric']}")
    if best_opts["metric"] == "minkowski":
        print(f"  Minkowski p-parameter: {best_opts['p']}")
    print(f"  Weighting: {best_opts['weights']}")
    print(f"  Algorithm: {best_opts['algorithm']}")
    print(f"  Leaf Size: {best_opts['leaf_size']}")

    # Performance insights
    n_samples, n_features = X_train.shape
    print(f"\n💡 Performance Insights:")

    k = best_opts["n_neighbors"]
    if k == 1:
        print(f"  ⚠️  k=1 suggests overfitting or very clean data")
    elif k > np.sqrt(n_samples):
        print(f"  ⚠️  Large k ({k}) - might be smoothing too much")
    else:
        print(f"  ✅ k={k} is reasonable for dataset size")

    if best_opts["weights"] == "distance":
        print(f"  ✅ Distance weighting helps with varying neighbor relevance")

    if best_opts["metric"] == "cosine":
        print(f"  💡 Cosine metric suggests directional relationships matter")
    elif best_opts["metric"] == "manhattan":
        print(f"  💡 Manhattan metric good for sparse/noisy data")

    # Scalability analysis
    print(f"\n⚡ Scalability Analysis:")
    print(f"  Training Complexity: O(1) - just stores data")
    print(f"  Prediction Complexity: O(n*d) per query")
    print(f"  Memory Usage: O(n*d) - stores all training data")

    if n_samples > 10000:
        print(f"  ⚠️  Large dataset ({n_samples}) - consider approximate methods")
    if n_features > 50:
        print(f"  ⚠️  High dimensions ({n_features}) - curse of dimensionality")
    if pred_time > 1:
        print(f"  ⚠️  Slow predictions ({pred_time:.1f}s) - consider faster algorithms")

    # Recommendations
    print(f"\n🚀 Recommendations:")
    if accuracy < 0.7 and n_features > 20:
        print(f"  💡 Consider dimensionality reduction (PCA, feature selection)")
    if pred_time > 5:
        print(f"  💡 Consider approximate methods (LSH) or different algorithms")
    if n_samples > 50000:
        print(f"  💡 Consider faster algorithms (XGBoost, RandomForest) for production")

    return {
        "accuracy": accuracy,
        "f1": f1,
        "training_time": fit_time,
        "prediction_time": pred_time,
        "baseline_accuracy": baseline_acc,
        "baseline_f1": baseline_f1,
        "accuracy_improvement": accuracy - baseline_acc,
        "f1_improvement": f1 - baseline_f1,
        "optimal_k": best_opts["n_neighbors"],
        "optimal_metric": best_opts["metric"],
    }


# Example usage function
def example_usage():
    """Example of FAST KNN optimization"""
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    print("🚀 FAST KNN Optimization")
    print("Focus: Smart Parameter Selection + Speed")
    print("=" * 50)

    # Generate sample data (keep it moderate size for KNN)
    X, y = make_classification(
        n_samples=2000,  # Moderate size for KNN demo
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        random_state=42,
    )

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale the data (CRITICAL for KNN!)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(
        f"Dataset: {X_train_scaled.shape[0]} samples, {X_train_scaled.shape[1]} features"
    )
    print("🔥 Feature scaling is CRITICAL for KNN performance!")

    # Run FAST optimization
    best_opts, best_acc, best_f1, acc_history, f1_history = _optimize_knn(
        X_train_scaled, y_train, X_test_scaled, y_test, cycles=1  # Start with 1 cycle
    )

    print(f"\n🎯 OPTIMIZATION COMPLETE!")
    print(f"Best KNN Accuracy: {best_acc:.4f}")
    print(f"Best KNN F1: {best_f1:.4f}")
    print(f"Optimal k: {best_opts['n_neighbors']}")
    print(f"Best Metric: {best_opts['metric']}")

    # Analyze performance
    analysis = _analyze_knn_performance(
        X_train_scaled, y_train, X_test_scaled, y_test, best_opts
    )

    return best_opts, best_acc, best_f1, acc_history, f1_history
