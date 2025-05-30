from typing import Any
import warnings
import numpy as np
import time
from tqdm.auto import tqdm
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import (
    RBF,
    Matern,
    RationalQuadratic,
    ExpSineSquared,
    DotProduct,
    WhiteKernel,
    ConstantKernel,
)
from sklearn.metrics import accuracy_score, f1_score
from sklearn.exceptions import ConvergenceWarning
import psutil

# Suppress convergence warnings for cleaner progress bars
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Global efficiency controls for GP (very aggressive due to O(n^3) scaling)
_max_time_per_step = 900  # 15 minutes max per optimization step
_max_time_per_model = 300  # 5 minutes max per model evaluation
_min_accuracy_threshold = 0.15  # Stop if accuracy is consistently terrible
_consecutive_failures = 0
_max_consecutive_failures = 2  # Very aggressive for GPs

# GP scalability limits
_max_recommended_samples = 1000  # Beyond this, GP becomes impractical
_max_safe_samples = 2000  # Absolute maximum before likely crash
_min_memory_gb = 4  # Minimum RAM for GP


def _check_gp_suitability(X_train, y_train):
    """Check if dataset is suitable for Gaussian Process"""
    n_samples, n_features = X_train.shape
    n_classes = len(np.unique(y_train))

    # Memory check
    available_memory_gb = psutil.virtual_memory().available / (1024**3)

    print("🔍 Gaussian Process Suitability Check:")
    print("=" * 50)
    print(f"Dataset: {n_samples} samples, {n_features} features, {n_classes} classes")
    print(f"Available RAM: {available_memory_gb:.1f} GB")

    # Estimate memory requirements (rough approximation)
    estimated_memory_gb = (n_samples**2 * 8) / (1024**3)  # 8 bytes per float64
    print(f"Estimated GP memory: {estimated_memory_gb:.1f} GB")

    # Check suitability
    suitable = True
    warnings_issued = []

    if n_samples > _max_safe_samples:
        print(
            f"❌ CRITICAL: Dataset too large ({n_samples} > {_max_safe_samples} samples)"
        )
        print("Gaussian Process WILL likely crash or take days to complete.")
        print("STRONG RECOMMENDATION: Use different algorithms:")
        print("  - XGBoost, RandomForest, or Neural Networks for large data")
        print("  - Consider subsampling to < 500 samples if GP is required")
        return False

    elif n_samples > _max_recommended_samples:
        print(
            f"⚠️  WARNING: Large dataset ({n_samples} > {_max_recommended_samples} samples)"
        )
        print("Gaussian Process will be very slow (hours).")
        print("Consider alternatives or subsampling.")
        warnings_issued.append("large_dataset")

    elif n_samples > 500:
        print(f"⚠️  WARNING: Medium dataset ({n_samples} samples)")
        print("Gaussian Process will be slow (10-60+ minutes).")
        warnings_issued.append("medium_dataset")

    else:
        print(f"✅ Good dataset size ({n_samples} samples) for GP")

    if estimated_memory_gb > available_memory_gb * 0.8:
        print(f"❌ MEMORY WARNING: GP may use {estimated_memory_gb:.1f} GB")
        print(f"Available: {available_memory_gb:.1f} GB - risk of crash!")
        if estimated_memory_gb > available_memory_gb:
            return False
        warnings_issued.append("high_memory")

    if n_features > 50:
        print(f"⚠️  WARNING: High-dimensional data ({n_features} features)")
        print("GP may struggle with curse of dimensionality.")
        print("Consider dimensionality reduction (PCA).")
        warnings_issued.append("high_dimension")

    if n_classes > 5:
        print(f"⚠️  WARNING: Many classes ({n_classes})")
        print("GP multi-class is expensive. Consider binary classification.")
        warnings_issued.append("many_classes")

    # Performance estimates
    if n_samples <= 100:
        print("⏱️  Expected time: 30 seconds - 2 minutes")
    elif n_samples <= 500:
        print("⏱️  Expected time: 2-15 minutes")
    elif n_samples <= 1000:
        print("⏱️  Expected time: 15-60 minutes")
    else:
        print("⏱️  Expected time: 1+ hours (possibly much longer)")

    print("\n💡 GP is best for:")
    print("  - Small, high-quality datasets (< 500 samples)")
    print("  - Problems where uncertainty quantification is important")
    print("  - Non-linear patterns with limited data")
    print("  - Scientific applications requiring interpretability")

    if len(warnings_issued) > 2:
        print(f"\n⚠️  RECOMMENDATION: Consider faster alternatives:")
        print("  - XGBoost or RandomForest for most problems")
        print("  - Neural Networks for complex patterns")
        print("  - SGD or LogisticRegression for large datasets")

    return suitable


def _safe_evaluate_model(X_train, y_train, X_test, y_true, opts):
    """Safely evaluate a GP model configuration with timeout protection"""
    global _consecutive_failures

    try:
        start_time = time.time()
        classifier = GaussianProcessClassifier(**opts)

        # Check memory before fitting
        memory_before = psutil.virtual_memory().percent

        classifier.fit(X_train, y_train)

        # Check if training took too long
        training_time = time.time() - start_time
        if training_time > _max_time_per_model:
            print(f"⏰ GP timeout after {training_time:.1f}s")
            return 0.0, 0.0, False

        # Check memory usage
        memory_after = psutil.virtual_memory().percent
        if memory_after > 90:
            print(f"🧠 High memory usage: {memory_after:.1f}%")

        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        # Track consecutive failures
        if accuracy < _min_accuracy_threshold:
            _consecutive_failures += 1
        else:
            _consecutive_failures = 0

        return accuracy, f1, True

    except MemoryError:
        print("🧠 OUT OF MEMORY - GP crashed!")
        print("Dataset too large for available RAM.")
        _consecutive_failures += 1
        return 0.0, 0.0, False

    except Exception as e:
        print(f"GP evaluation failed: {e}")
        _consecutive_failures += 1
        return 0.0, 0.0, False


def _get_smart_kernels(n_samples, n_features):
    """Get smart kernel selection based on dataset characteristics"""

    kernels = {}

    # Always include RBF (most common)
    kernels["rbf"] = 1.0 * RBF(1.0)

    # Add Matern for small datasets (more flexible than RBF)
    if n_samples < 500:
        kernels["matern_1.5"] = 1.0 * Matern(length_scale=1.0, nu=1.5)
        kernels["matern_2.5"] = 1.0 * Matern(length_scale=1.0, nu=2.5)

    # Add noise kernel for noisy data
    if n_samples < 1000:  # Only for smaller datasets due to computational cost
        kernels["rbf_white"] = 1.0 * RBF(1.0) + WhiteKernel(noise_level=1e-5)

    # Add RationalQuadratic for complex patterns (small datasets only)
    if n_samples < 300:
        kernels["rational_quadratic"] = 1.0 * RationalQuadratic(
            length_scale=1.0, alpha=1.0
        )

    # DotProduct for linear-ish relationships
    if n_features < 20:  # Works better in lower dimensions
        kernels["dot_product"] = DotProduct(sigma_0=1.0)

    return kernels


def _optimize_kernel_type(X_train, y_train, X_test, y_true, opts):
    """Optimize kernel type with dataset-aware selection"""
    global _consecutive_failures
    start_time = time.time()
    _consecutive_failures = 0

    max_acc = -np.inf
    best_f1 = 0.0
    n_samples, n_features = X_train.shape

    # Get smart kernels based on dataset characteristics
    kernels = _get_smart_kernels(n_samples, n_features)

    best_kernel = "rbf"
    kernel_items = list(kernels.items())

    with tqdm(kernel_items, desc="Optimizing Kernel Type", leave=False) as pbar:
        for kernel_name, kernel in pbar:
            # Early stopping conditions
            if time.time() - start_time > _max_time_per_step:
                pbar.set_description("Kernel Type (TIME LIMIT)")
                break
            if _consecutive_failures >= _max_consecutive_failures:
                pbar.set_description("Kernel Type (POOR ACCURACY)")
                break

            test_opts = opts.copy()
            test_opts["kernel"] = kernel

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_kernel = kernel_name
                opts["kernel"] = kernel

            pbar.set_postfix(
                {
                    "current": kernel_name[:8],
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                    "best": best_kernel[:8],
                }
            )

    return opts, max_acc, best_f1


def _optimize_key_hyperparams(X_train, y_train, X_test, y_true, opts):
    """Optimize key hyperparameters together for efficiency"""
    global _consecutive_failures
    start_time = time.time()
    _consecutive_failures = 0

    max_acc = -np.inf
    best_f1 = 0.0
    n_samples = X_train.shape[0]

    # Smart parameter combinations based on dataset size
    if n_samples < 200:
        # Small dataset - can afford more restarts and iterations
        configs = [
            {"n_restarts_optimizer": 2, "max_iter_predict": 200},
            {"n_restarts_optimizer": 5, "max_iter_predict": 200},
            {"n_restarts_optimizer": 1, "max_iter_predict": 100},
            {"n_restarts_optimizer": 0, "max_iter_predict": 100},
        ]
    elif n_samples < 500:
        # Medium dataset - balanced approach
        configs = [
            {"n_restarts_optimizer": 1, "max_iter_predict": 100},
            {"n_restarts_optimizer": 2, "max_iter_predict": 100},
            {"n_restarts_optimizer": 0, "max_iter_predict": 200},
        ]
    else:
        # Large dataset - minimize computational cost
        configs = [
            {"n_restarts_optimizer": 0, "max_iter_predict": 100},
            {"n_restarts_optimizer": 1, "max_iter_predict": 100},
        ]

    best_config = configs[0]

    with tqdm(configs, desc="Optimizing Key Hyperparams", leave=False) as pbar:
        for config in pbar:
            # Early stopping conditions
            if time.time() - start_time > _max_time_per_step:
                pbar.set_description("Key Hyperparams (TIME LIMIT)")
                break
            if _consecutive_failures >= _max_consecutive_failures:
                pbar.set_description("Key Hyperparams (POOR ACCURACY)")
                break

            test_opts = opts.copy()
            test_opts.update(config)

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_config = config

            pbar.set_postfix(
                {
                    "restarts": config["n_restarts_optimizer"],
                    "max_iter": config["max_iter_predict"],
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts.update(best_config)
    return opts, max_acc, best_f1


def _optimize_length_scale(X_train, y_train, X_test, y_true, opts):
    """Optimize kernel length scale with reduced search space"""
    global _consecutive_failures
    start_time = time.time()
    _consecutive_failures = 0

    max_acc = -np.inf
    best_f1 = 0.0
    n_samples = X_train.shape[0]

    # Reduced search space for speed
    if n_samples < 200:
        variable_array = np.logspace(-1, 1, 5)  # 5 values instead of 10
    else:
        variable_array = np.logspace(-1, 1, 3)  # Only 3 values for larger datasets

    current_kernel = opts["kernel"]

    with tqdm(variable_array, desc="Optimizing Length Scale", leave=False) as pbar:
        for length_scale in pbar:
            # Early stopping conditions
            if time.time() - start_time > _max_time_per_step:
                pbar.set_description("Length Scale (TIME LIMIT)")
                break
            if _consecutive_failures >= _max_consecutive_failures:
                pbar.set_description("Length Scale (POOR ACCURACY)")
                break

            new_kernel = _create_kernel_with_length_scale(current_kernel, length_scale)

            if new_kernel is None:
                pbar.set_postfix({"scale": f"{length_scale:.3f}", "status": "skipped"})
                continue

            test_opts = opts.copy()
            test_opts["kernel"] = new_kernel

            accuracy, f1, success = _safe_evaluate_model(
                X_train, y_train, X_test, y_true, test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                opts["kernel"] = new_kernel

            pbar.set_postfix(
                {
                    "scale": f"{length_scale:.3f}",
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best": f"{max_acc:.4f}",
                }
            )

    return opts, max_acc, best_f1


def _create_kernel_with_length_scale(kernel, length_scale):
    """Create new kernel with updated length scale"""
    try:
        # Handle composite kernels (with WhiteKernel)
        if hasattr(kernel, "k1") and hasattr(kernel, "k2"):
            # Find the base kernel and noise kernel
            if isinstance(kernel.k1, (RBF, Matern, RationalQuadratic)):
                base_kernel = kernel.k1
                noise_kernel = kernel.k2
            elif isinstance(kernel.k2, (RBF, Matern, RationalQuadratic)):
                base_kernel = kernel.k2
                noise_kernel = kernel.k1
            else:
                return None

            new_base = _create_simple_kernel_with_scale(base_kernel, length_scale)
            return new_base + noise_kernel if new_base else None

        # Handle simple kernels
        return _create_simple_kernel_with_scale(kernel, length_scale)

    except Exception:
        return None


def _create_simple_kernel_with_scale(kernel, length_scale):
    """Create simple kernel with specified length scale"""
    if isinstance(kernel, RBF):
        return RBF(length_scale=length_scale)
    elif isinstance(kernel, Matern):
        return Matern(length_scale=length_scale, nu=kernel.nu)
    elif isinstance(kernel, RationalQuadratic):
        return RationalQuadratic(length_scale=length_scale, alpha=kernel.alpha)
    return None


def _optimize_gaussian_process(X_train, y_train, X_test, y_true, cycles=1):
    """
    SAFE optimized hyperparameters for GaussianProcessClassifier.
    :param X_train: Training data (MUST be small - GP scales O(n^3)!)
    :param y_train: Training labels
    :param X_test: Test data
    :param y_true: True labels for the test data
    :param cycles: Number of optimization cycles (keep low for GP!)
    :return: optimized options, best accuracy, best f1, accuracy history, f1 history
    """

    print("🧠 SAFE Gaussian Process Optimization")
    print("CRITICAL: GP scales O(n³) - only for small datasets!")
    print("=" * 60)

    # Critical suitability check
    if not _check_gp_suitability(X_train, y_train):
        print("\n❌ STOPPING: Dataset unsuitable for Gaussian Process!")
        print("Returning dummy results. Please use different algorithms.")
        dummy_opts = {"kernel": RBF(1.0), "n_restarts_optimizer": 0}
        return dummy_opts, 0.0, 0.0, [0.0], [0.0]

    n_samples, n_features = X_train.shape
    n_classes = len(np.unique(y_train))

    # Quick baseline check
    print("\nRunning baseline check...")
    baseline_start = time.time()

    try:
        baseline_gp = GaussianProcessClassifier(
            kernel=RBF(1.0), n_restarts_optimizer=0, random_state=42
        )
        baseline_gp.fit(X_train, y_train)
        baseline_acc = baseline_gp.score(X_test, y_true)
        baseline_time = time.time() - baseline_start

        print(f"Baseline GP: {baseline_acc:.4f} (trained in {baseline_time:.1f}s)")

        if baseline_time > 60:
            print("⚠️  Slow baseline - full optimization will take significant time")

    except Exception as e:
        print(f"Baseline check failed: {e}")
        baseline_acc = 0.0
        baseline_time = 0

    # Define initial parameters with safe defaults
    opts = {
        "kernel": 1.0 * RBF(1.0),
        "optimizer": "fmin_l_bfgs_b",
        "n_restarts_optimizer": 0,  # Start conservative
        "max_iter_predict": 100,  # Start conservative
        "warm_start": False,
        "copy_X_train": True,
        "random_state": 42,
        "multi_class": "one_vs_rest" if n_classes > 2 else "one_vs_one",
        "n_jobs": 1,  # GP doesn't parallelize well, can cause issues
    }

    # Track results
    ma_vec = []
    f1_vec = []

    # Main optimization loop (keep cycles low for GP!)
    with tqdm(range(cycles), desc="SAFE GP Optimization", position=0) as cycle_pbar:
        for c in cycle_pbar:
            cycle_start_time = time.time()
            cycle_pbar.set_description(f"GP Cycle {c + 1}/{cycles}")

            # Core optimizations (minimal for speed/safety)
            opts, _, _ = _optimize_kernel_type(X_train, y_train, X_test, y_true, opts)
            opts, _, _ = _optimize_key_hyperparams(
                X_train, y_train, X_test, y_true, opts
            )

            # Only optimize length scale for small datasets
            if n_samples < 500:
                opts, ma, f1 = _optimize_length_scale(
                    X_train, y_train, X_test, y_true, opts
                )
            else:
                # Skip length scale optimization for larger datasets
                ma, f1, _ = _safe_evaluate_model(X_train, y_train, X_test, y_true, opts)

            ma_vec.append(ma)
            f1_vec.append(f1)

            cycle_time = time.time() - cycle_start_time

            # Extract kernel info for display
            kernel_name = "Unknown"
            if hasattr(opts["kernel"], "k1"):
                kernel_name = type(opts["kernel"].k1).__name__
            else:
                kernel_name = type(opts["kernel"]).__name__

            cycle_pbar.set_postfix(
                {
                    "accuracy": f"{ma:.4f}",
                    "f1": f"{f1:.4f}",
                    "best_overall": f"{max(ma_vec):.4f}",
                    "cycle_time": f"{cycle_time:.1f}s",
                    "kernel": kernel_name[:8],
                    "restarts": opts["n_restarts_optimizer"],
                    "baseline_beat": f"{ma - baseline_acc:+.4f}",
                }
            )

    total_time = time.time() - baseline_start
    print(f"\n🎯 GP optimization completed in {total_time:.1f}s total")

    return opts, ma, f1, ma_vec, f1_vec


def _analyze_gp_performance(X_train, y_train, X_test, y_true, best_opts):
    """Analyze GP performance and provide insights"""

    print("\n" + "=" * 70)
    print("SAFE GAUSSIAN PROCESS PERFORMANCE ANALYSIS")
    print("=" * 70)

    # Train final model
    print("Training final GP with best parameters...")
    start_time = time.time()

    try:
        gp_clf = GaussianProcessClassifier(**best_opts)
        gp_clf.fit(X_train, y_train)
        training_time = time.time() - start_time

        y_pred = gp_clf.predict(X_test)
        y_pred_proba = gp_clf.predict_proba(X_test)

        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted")

        # Calculate prediction confidence
        max_proba = np.max(y_pred_proba, axis=1)
        mean_confidence = np.mean(max_proba)

        training_success = True

    except Exception as e:
        print(f"Final training failed: {e}")
        accuracy = f1 = training_time = mean_confidence = 0.0
        training_success = False

    if training_success:
        print(f"\n🎯 Final GP Performance:")
        print(f"  Test Accuracy: {accuracy:.4f}")
        print(f"  Test F1 Score: {f1:.4f}")
        print(f"  Training Time: {training_time:.1f} seconds")
        print(f"  Mean Prediction Confidence: {mean_confidence:.4f}")

        # Kernel analysis
        print(f"\n🔧 Optimal Configuration:")
        kernel = best_opts["kernel"]

        if hasattr(kernel, "k1") and hasattr(kernel, "k2"):
            print(
                f"  Kernel: Composite ({type(kernel.k1).__name__} + {type(kernel.k2).__name__})"
            )
        else:
            print(f"  Kernel: {type(kernel).__name__}")

        if hasattr(kernel, "length_scale"):
            print(f"  Length Scale: {kernel.length_scale:.4f}")
        elif hasattr(kernel, "k1") and hasattr(kernel.k1, "length_scale"):
            print(f"  Length Scale: {kernel.k1.length_scale:.4f}")

        print(f"  Optimizer Restarts: {best_opts['n_restarts_optimizer']}")
        print(f"  Max Iter Predict: {best_opts['max_iter_predict']}")

        # Dataset analysis
        n_samples, n_features = X_train.shape
        print(f"\n📊 Dataset Characteristics:")
        print(f"  Training Samples: {n_samples}")
        print(f"  Features: {n_features}")
        print(f"  Classes: {len(np.unique(y_train))}")

        # Complexity analysis
        print(f"\n🧮 Computational Complexity:")
        print(f"  Training: O(n³) = O({n_samples}³) ≈ {n_samples ** 3:,} operations")
        print(f"  Prediction: O(n) per sample")
        print(f"  Memory: O(n²) ≈ {(n_samples ** 2 * 8) / (1024 ** 2):.1f} MB")

        # Performance insights
        print(f"\n💡 Performance Insights:")

        if accuracy > 0.9:
            print(f"  ✅ Excellent performance - GP is working very well")
        elif accuracy > 0.8:
            print(f"  ✅ Good performance - GP captures the pattern well")
        elif accuracy > 0.6:
            print(f"  ⚠️  Moderate performance - consider kernel tuning or more data")
        else:
            print(f"  ❌ Poor performance - GP may not be suitable for this problem")

        if mean_confidence > 0.8:
            print(f"  ✅ High prediction confidence - GP is certain about predictions")
        elif mean_confidence > 0.6:
            print(f"  ⚠️  Moderate confidence - some uncertainty in predictions")
        else:
            print(f"  ❌ Low confidence - high uncertainty, consider more data")

        if training_time > 300:  # 5 minutes
            print(f"  ⏰ Long training time - dataset pushing GP limits")

        # Recommendations
        print(f"\n🚀 Recommendations:")

        if n_samples > 500:
            print(f"  💡 Consider subsampling for faster training")
        if n_features > 20:
            print(f"  💡 Consider PCA for dimensionality reduction")
        if accuracy < 0.7:
            print(f"  💡 GP may not be best choice - try XGBoost or RandomForest")
        if training_time > 60:
            print(f"  💡 For production, consider approximation methods (sparse GP)")

        print(f"\n🎯 GP Advantages for this problem:")
        print(f"  ✅ Provides uncertainty quantification")
        print(f"  ✅ Works well with limited data")
        print(f"  ✅ Handles non-linear patterns")
        print(f"  ✅ No hyperparameter tuning needed (marginal likelihood)")

    return {
        "accuracy": accuracy,
        "f1": f1,
        "training_time": training_time,
        "mean_confidence": mean_confidence,
        "training_success": training_success,
        "n_samples": X_train.shape[0],
        "kernel_type": type(best_opts["kernel"]).__name__,
    }


# Example usage function
def example_usage():
    """Example of SAFE GP optimization"""
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    print("🚀 SAFE Gaussian Process Optimization")
    print("Focus: Dataset Suitability + Safety")
    print("=" * 50)

    # Generate SMALL sample data (GP requirement!)
    print("Generating SMALL sample data for GP...")

    X, y = make_classification(
        n_samples=300,  # Small for GP!
        n_features=10,  # Moderate features
        n_informative=8,
        n_redundant=2,
        n_classes=3,
        random_state=42,
    )

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Scale the data (important for GP!)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(
        f"Dataset: {X_train_scaled.shape[0]} samples, {X_train_scaled.shape[1]} features"
    )
    print("🔥 Small dataset size is CRITICAL for GP performance!")

    # Run SAFE GP optimization
    best_opts, best_acc, best_f1, acc_history, f1_history = _optimize_gaussian_process(
        X_train_scaled,
        y_train,
        X_test_scaled,
        y_test,
        cycles=1,  # Keep cycles=1 for safety
    )

    print(f"\n🎯 OPTIMIZATION COMPLETE!")
    print(f"Best GP Accuracy: {best_acc:.4f}")
    print(f"Best GP F1: {best_f1:.4f}")

    # Analyze performance
    analysis = _analyze_gp_performance(
        X_train_scaled, y_train, X_test_scaled, y_test, best_opts
    )

    return best_opts, best_acc, best_f1, acc_history, f1_history
