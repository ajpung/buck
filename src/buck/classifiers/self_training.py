import warnings
import numpy as np
import time
from tqdm.auto import tqdm
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
import psutil

try:
    import xgboost as xgb

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# Robust efficiency controls
_max_time_per_step = 600  # 10 minutes max per optimization step
_max_time_per_model = 900  # 5 minutes max per model evaluation
_min_accuracy_threshold = 0.15  # Stop if accuracy is consistently terrible
_consecutive_failures = 0
_max_consecutive_failures = 3  # More tolerant

# Dataset limits for self-training
_max_recommended_samples = 10000
_max_safe_samples = 50000


def _create_default_return_values():
    """Create safe default return values for error cases"""
    default_opts = {
        "base_estimator": LogisticRegression(random_state=42, max_iter=100),
        "threshold": 0.75,
        "criterion": "threshold",
        "k_best": 10,
        "max_iter": 10,
        "verbose": False,
    }
    return default_opts, 0.0, 0.0, [0.0], [0.0]


def _check_self_training_suitability(X_train, y_train):
    """Check if dataset is suitable for self-training with robust error handling"""
    try:
        n_samples, n_features = X_train.shape
        n_classes = len(np.unique(y_train))

        # Memory check with error handling
        try:
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
        except Exception:
            available_memory_gb = 8.0  # Default assumption

        print("🔍 Self-Training Suitability Check:")
        print("=" * 50)
        print(
            f"Dataset: {n_samples} samples, {n_features} features, {n_classes} classes"
        )
        print(f"Available RAM: {available_memory_gb:.1f} GB")

        suitable = True
        warnings_issued = []

        if n_samples > _max_safe_samples:
            print(
                f"❌ CRITICAL: Dataset too large ({n_samples} > {_max_safe_samples} samples)"
            )
            print("Self-training will be extremely slow or crash.")
            print("RECOMMENDATION: Use simpler algorithms or subsample data")
            return False, ["dataset_too_large"]

        elif n_samples > _max_recommended_samples:
            print(
                f"⚠️  WARNING: Large dataset ({n_samples} > {_max_recommended_samples} samples)"
            )
            print("Self-training will be slow (30+ minutes).")
            warnings_issued.append("large_dataset")

        elif n_samples < 100:
            print(f"⚠️  WARNING: Very small dataset ({n_samples} samples)")
            print("Self-training may not be beneficial.")
            warnings_issued.append("small_dataset")
        else:
            print(f"✅ Good dataset size ({n_samples} samples) for self-training")

        if n_features > 1000:
            print(f"⚠️  WARNING: High-dimensional data ({n_features} features)")
            print("Consider dimensionality reduction before self-training.")
            warnings_issued.append("high_dimension")

        if n_classes > 10:
            print(f"⚠️  WARNING: Many classes ({n_classes})")
            print("Self-training may struggle with highly multi-class problems.")
            warnings_issued.append("many_classes")

        return suitable, warnings_issued

    except Exception as e:
        print(f"❌ Error in suitability check: {e}")
        print("Proceeding with caution...")
        return True, ["suitability_check_failed"]


def _safe_evaluate_self_training(
    X_train, y_train, X_test, y_true, base_estimator, **kwargs
):
    """Safely evaluate a self-training configuration with robust error handling"""
    global _consecutive_failures

    try:
        start_time = time.time()

        # Create self-training classifier with error handling
        try:
            st_classifier = SelfTrainingClassifier(
                base_estimator=base_estimator, **kwargs
            )
        except Exception as e:
            print(f"❌ Failed to create SelfTrainingClassifier: {e}")
            _consecutive_failures += 1
            return 0.0, 0.0, False

        # Check memory before fitting
        try:
            memory_before = psutil.virtual_memory().percent
        except Exception:
            memory_before = 50  # Default assumption

        # Fit the model with timeout protection
        try:
            st_classifier.fit(X_train, y_train)
        except Exception as e:
            print(f"❌ Self-training fit failed: {e}")
            _consecutive_failures += 1
            return 0.0, 0.0, False

        # Check if training took too long
        training_time = time.time() - start_time
        if training_time > _max_time_per_model:
            print(f"⏰ Self-training timeout after {training_time:.1f}s")
            _consecutive_failures += 1
            return 0.0, 0.0, False

        # Check memory usage
        try:
            memory_after = psutil.virtual_memory().percent
            if memory_after > 90:
                print(f"🧠 High memory usage: {memory_after:.1f}%")
        except Exception:
            pass  # Memory check failed, continue

        # Make predictions with error handling
        try:
            y_pred = st_classifier.predict(X_test)
            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            _consecutive_failures += 1
            return 0.0, 0.0, False

        # Track consecutive failures
        if accuracy < _min_accuracy_threshold:
            _consecutive_failures += 1
        else:
            _consecutive_failures = 0

        return accuracy, f1, True

    except Exception as e:
        print(f"❌ Unexpected error in self-training evaluation: {e}")
        _consecutive_failures += 1
        return 0.0, 0.0, False


def _get_smart_base_estimators(n_samples, n_features, n_classes):
    """Get smart base estimator selection with robust error handling"""

    estimators = {}

    try:
        # Always include fast, reliable estimators
        estimators["logistic"] = LogisticRegression(
            random_state=42,
            max_iter=1000,
            solver="liblinear" if n_samples < 10000 else "lbfgs",
        )

        estimators["gaussian_nb"] = GaussianNB()

        # Add SGD for larger datasets
        if n_samples > 1000:
            estimators["sgd"] = SGDClassifier(
                random_state=42,
                max_iter=1000,
                loss="log_loss" if n_classes > 2 else "hinge",
            )

        # Add tree-based for smaller datasets
        if n_samples < 5000:
            estimators["random_forest"] = RandomForestClassifier(
                n_estimators=50,  # Smaller for speed
                max_depth=10,
                random_state=42,
                n_jobs=1,
            )

            estimators["extra_trees"] = ExtraTreesClassifier(
                n_estimators=50, max_depth=10, random_state=42, n_jobs=1
            )

        # Add XGBoost if available and dataset not too large
        if HAS_XGB and n_samples < 10000:
            try:
                estimators["xgboost"] = xgb.XGBClassifier(
                    n_estimators=50,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=1,
                    verbosity=0,
                )
            except Exception as e:
                print(f"⚠️  XGBoost creation failed: {e}")

        # Add KNN for very small datasets
        if n_samples < 2000 and n_features < 50:
            estimators["knn"] = KNeighborsClassifier(
                n_neighbors=min(5, n_samples // 10), n_jobs=1
            )

        # Add SVM for small datasets
        if n_samples < 1000:
            try:
                estimators["svm"] = SVC(
                    kernel="rbf",
                    probability=True,  # Needed for self-training
                    random_state=42,
                )
            except Exception as e:
                print(f"⚠️  SVM creation failed: {e}")

    except Exception as e:
        print(f"❌ Error creating base estimators: {e}")
        # Return minimal safe estimator
        estimators = {"logistic": LogisticRegression(random_state=42, max_iter=100)}

    return estimators


def _optimize_base_estimator(X_train, y_train, X_test, y_true, **kwargs):
    """Optimize base estimator with robust error handling"""
    global _consecutive_failures
    start_time = time.time()
    _consecutive_failures = 0

    max_acc = -np.inf
    best_f1 = 0.0
    best_estimator_name = "logistic"
    best_estimator = None

    try:
        n_samples, n_features = X_train.shape
        n_classes = len(np.unique(y_train))

        base_estimators = _get_smart_base_estimators(n_samples, n_features, n_classes)

        if not base_estimators:
            print("❌ No base estimators available!")
            return LogisticRegression(random_state=42), 0.0, 0.0

        with tqdm(
            base_estimators.items(), desc="Optimizing Base Estimator", leave=False
        ) as pbar:
            for name, estimator in pbar:
                # Early stopping conditions
                if time.time() - start_time > _max_time_per_step:
                    pbar.set_description("Base Estimator (TIME LIMIT)")
                    break
                if _consecutive_failures >= _max_consecutive_failures:
                    pbar.set_description("Base Estimator (TOO MANY FAILURES)")
                    break

                accuracy, f1, success = _safe_evaluate_self_training(
                    X_train, y_train, X_test, y_true, estimator, **kwargs
                )

                if success and accuracy >= max_acc:
                    max_acc = accuracy
                    best_f1 = f1
                    best_estimator_name = name
                    best_estimator = estimator

                pbar.set_postfix(
                    {
                        "estimator": name[:10],
                        "acc": f"{accuracy:.4f}" if success else "failed",
                        "best_acc": f"{max_acc:.4f}",
                        "best": best_estimator_name[:10],
                    }
                )

        # Ensure we return a valid estimator
        if best_estimator is None:
            print("⚠️  No successful base estimator, using LogisticRegression")
            best_estimator = LogisticRegression(random_state=42, max_iter=100)
            max_acc = 0.0
            best_f1 = 0.0

        return best_estimator, max_acc, best_f1

    except Exception as e:
        print(f"❌ Error in base estimator optimization: {e}")
        return LogisticRegression(random_state=42, max_iter=100), 0.0, 0.0


def _optimize_self_training_params(X_train, y_train, X_test, y_true, best_estimator):
    """Optimize self-training parameters with robust error handling"""
    global _consecutive_failures
    start_time = time.time()
    _consecutive_failures = 0

    max_acc = -np.inf
    best_f1 = 0.0
    best_params = {}

    try:
        n_samples = X_train.shape[0]

        # Smart parameter combinations based on dataset size
        if n_samples < 1000:
            param_combinations = [
                {
                    "threshold": 0.75,
                    "criterion": "threshold",
                    "k_best": 10,
                    "max_iter": 10,
                },
                {
                    "threshold": 0.80,
                    "criterion": "threshold",
                    "k_best": 20,
                    "max_iter": 20,
                },
                {
                    "threshold": 0.70,
                    "criterion": "threshold",
                    "k_best": 5,
                    "max_iter": 5,
                },
            ]
        elif n_samples < 5000:
            param_combinations = [
                {
                    "threshold": 0.75,
                    "criterion": "threshold",
                    "k_best": 50,
                    "max_iter": 10,
                },
                {
                    "threshold": 0.80,
                    "criterion": "threshold",
                    "k_best": 100,
                    "max_iter": 10,
                },
            ]
        else:
            # Minimal testing for large datasets
            param_combinations = [
                {
                    "threshold": 0.75,
                    "criterion": "threshold",
                    "k_best": 100,
                    "max_iter": 5,
                },
            ]

        with tqdm(param_combinations, desc="Optimizing ST Params", leave=False) as pbar:
            for params in pbar:
                # Early stopping conditions
                if time.time() - start_time > _max_time_per_step:
                    pbar.set_description("ST Params (TIME LIMIT)")
                    break
                if _consecutive_failures >= _max_consecutive_failures:
                    pbar.set_description("ST Params (TOO MANY FAILURES)")
                    break

                accuracy, f1, success = _safe_evaluate_self_training(
                    X_train, y_train, X_test, y_true, best_estimator, **params
                )

                if success and accuracy >= max_acc:
                    max_acc = accuracy
                    best_f1 = f1
                    best_params = params.copy()

                pbar.set_postfix(
                    {
                        "threshold": params["threshold"],
                        "k_best": params["k_best"],
                        "max_iter": params["max_iter"],
                        "acc": f"{accuracy:.4f}" if success else "failed",
                        "best_acc": f"{max_acc:.4f}",
                    }
                )

        # Ensure we have valid parameters
        if not best_params:
            print("⚠️  No successful parameters, using defaults")
            best_params = {
                "threshold": 0.75,
                "criterion": "threshold",
                "k_best": 10,
                "max_iter": 10,
            }
            max_acc = 0.0
            best_f1 = 0.0

        return best_params, max_acc, best_f1

    except Exception as e:
        print(f"❌ Error in parameter optimization: {e}")
        default_params = {
            "threshold": 0.75,
            "criterion": "threshold",
            "k_best": 10,
            "max_iter": 10,
        }
        return default_params, 0.0, 0.0


def _optimize_selftrain(X_train, y_train, X_test, y_true, cycles=2):
    """
    ROBUST Self-Training Classifier optimization with guaranteed return values

    ALWAYS returns exactly 5 values: (opts, accuracy, f1, accuracy_history, f1_history)
    """

    print("🤖 ROBUST Self-Training Optimization")
    print("Focus: Reliability + Error Handling")
    print("=" * 60)

    try:
        # Critical suitability check
        suitable, warnings_list = _check_self_training_suitability(X_train, y_train)

        if not suitable:
            print("\n❌ Dataset unsuitable for self-training optimization!")
            print("Returning safe default values.")
            return _create_default_return_values()

        # Track results with guaranteed initialization
        ma_vec = []
        f1_vec = []

        # Initialize with safe defaults
        best_estimator = LogisticRegression(random_state=42, max_iter=100)
        best_params = {
            "threshold": 0.75,
            "criterion": "threshold",
            "k_best": 10,
            "max_iter": 10,
            "verbose": False,
        }
        final_accuracy = 0.0
        final_f1 = 0.0

        # Main optimization loop with error handling
        with tqdm(range(cycles), desc="Robust Self-Training", position=0) as cycle_pbar:
            for c in cycle_pbar:
                try:
                    cycle_start_time = time.time()
                    cycle_pbar.set_description(f"Self-Training Cycle {c + 1}/{cycles}")

                    # Step 1: Optimize base estimator
                    try:
                        best_estimator, _, _ = _optimize_base_estimator(
                            X_train, y_train, X_test, y_true, **best_params
                        )
                    except Exception as e:
                        print(f"❌ Base estimator optimization failed: {e}")
                        # Continue with default estimator

                    # Step 2: Optimize self-training parameters
                    try:
                        best_params, ma, f1 = _optimize_self_training_params(
                            X_train, y_train, X_test, y_true, best_estimator
                        )

                        # Update final results
                        final_accuracy = ma
                        final_f1 = f1

                    except Exception as e:
                        print(f"❌ Parameter optimization failed: {e}")
                        ma = f1 = 0.0

                    # Record results
                    ma_vec.append(ma)
                    f1_vec.append(f1)

                    cycle_time = time.time() - cycle_start_time

                    # Get estimator name safely
                    try:
                        estimator_name = type(best_estimator).__name__
                    except:
                        estimator_name = "Unknown"

                    cycle_pbar.set_postfix(
                        {
                            "accuracy": f"{ma:.4f}",
                            "f1": f"{f1:.4f}",
                            "best_overall": f"{max(ma_vec) if ma_vec else 0:.4f}",
                            "cycle_time": f"{cycle_time:.1f}s",
                            "estimator": estimator_name[:10],
                            "threshold": best_params.get("threshold", 0.75),
                        }
                    )

                except Exception as e:
                    print(f"❌ Cycle {c + 1} failed: {e}")
                    ma_vec.append(0.0)
                    f1_vec.append(0.0)
                    continue

        # Ensure we have valid histories
        if not ma_vec:
            ma_vec = [0.0]
        if not f1_vec:
            f1_vec = [0.0]

        # Create final options dictionary
        final_opts = {"base_estimator": best_estimator, **best_params}

        print(f"\n🎯 Robust self-training optimization completed!")
        print(f"Final accuracy: {final_accuracy:.4f}")
        print(f"Final F1: {final_f1:.4f}")

        # GUARANTEE: Always return exactly 5 values
        return final_opts, final_accuracy, final_f1, ma_vec, f1_vec

    except Exception as e:
        print(f"❌ CRITICAL ERROR in self-training optimization: {e}")
        print("Returning safe default values to prevent crash.")
        return _create_default_return_values()
