from typing import Any
import warnings
import numpy as np
import time
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score
import math

# Suppress warnings for cleaner progress bars
warnings.filterwarnings("ignore")

# Global efficiency controls - same as MLP
_max_time_per_step = 900  # 15 minutes max per optimization step (CNNs can be slow)
_max_time_per_model = 900  # 15 minutes max per model evaluation
_min_accuracy_threshold = 0.15  # Stop if accuracy is consistently terrible
_consecutive_failures = 0
_max_consecutive_failures = 3

# Try to import deep learning libraries
HAS_TENSORFLOW = False
HAS_TORCH = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    from tensorflow.keras.optimizers import Adam, SGD
    from tensorflow.keras.utils import to_categorical

    HAS_TENSORFLOW = True
    # Suppress TensorFlow warnings
    tf.get_logger().setLevel("ERROR")
except ImportError:
    pass

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    HAS_TORCH = True
except ImportError:
    pass


def _create_default_return_values():
    """Create safe default return values for error cases - same as other scripts"""
    default_opts = {
        "architecture": "simple",
        "optimizer": "adam",
        "learning_rate": 0.001,
        "epochs": 10,
        "batch_size": 32,
    }
    return default_opts, 0.0, 0.0, [0.0], [0.0]


def _check_cnn_suitability(X_train, y_train):
    """Check if dataset is suitable for CNN with robust error handling"""
    try:
        print("🔍 CNN Suitability Check:")
        print("=" * 50)

        original_shape = X_train.shape
        n_samples = original_shape[0]
        n_classes = len(np.unique(y_train))

        print(f"Original data shape: {original_shape}")
        print(f"Samples: {n_samples}, Classes: {n_classes}")

        # Check if we have deep learning libraries
        if not HAS_TENSORFLOW and not HAS_TORCH:
            print("❌ CRITICAL: No deep learning libraries found!")
            print("Need: pip install tensorflow OR pip install torch")
            print("Falling back to simple classification...")
            return False, "no_dl_libraries", original_shape

        # Determine data type and handle appropriately
        if len(original_shape) == 2:
            # 2D tabular data - needs reshaping for CNN
            n_features = original_shape[1]
            print(f"📊 Tabular data detected: {n_features} features")

            # Try to create reasonable image-like shape
            sqrt_features = int(math.sqrt(n_features))

            if sqrt_features * sqrt_features == n_features:
                # Perfect square - can reshape to square image
                new_shape = (n_samples, sqrt_features, sqrt_features, 1)
                print(
                    f"✅ Can reshape to square images: {sqrt_features}x{sqrt_features}"
                )
            else:
                # Find best rectangular shape
                best_h, best_w = _find_best_image_shape(n_features)
                if best_h * best_w <= n_features:
                    new_shape = (n_samples, best_h, best_w, 1)
                    print(f"✅ Can reshape to rectangular images: {best_h}x{best_w}")
                else:
                    print(
                        f"❌ Cannot create reasonable image shape from {n_features} features"
                    )
                    print("Recommendation: Use MLP/Neural Network instead of CNN")
                    return False, "unsuitable_shape", original_shape

            if sqrt_features < 8:
                print(
                    f"⚠️  WARNING: Very small images ({sqrt_features}x{sqrt_features})"
                )
                print("CNNs work better with larger images (32x32+)")

            return True, "tabular_data", new_shape

        elif len(original_shape) == 3:
            # Already 3D - assume grayscale images
            n_samples, height, width = original_shape
            print(f"🖼️  Grayscale images detected: {height}x{width}")

            if height < 8 or width < 8:
                print(f"⚠️  WARNING: Very small images ({height}x{width})")
                print("CNNs work better with larger images")

            # Add channel dimension
            new_shape = (n_samples, height, width, 1)
            return True, "grayscale_images", new_shape

        elif len(original_shape) == 4:
            # Already 4D - assume color images
            n_samples, height, width, channels = original_shape
            print(f"🖼️  Color images detected: {height}x{width}x{channels}")

            if height < 8 or width < 8:
                print(f"⚠️  WARNING: Very small images ({height}x{width})")

            return True, "color_images", original_shape

        else:
            print(f"❌ Unsupported data shape: {original_shape}")
            return False, "unsupported_shape", original_shape

    except Exception as e:
        print(f"❌ Error in CNN suitability check: {e}")
        return False, "check_failed", X_train.shape


def _find_best_image_shape(n_features):
    """Find best rectangular image shape for given number of features"""
    best_ratio = float("inf")
    best_h, best_w = 1, n_features

    for h in range(1, int(math.sqrt(n_features)) + 1):
        if n_features % h == 0:
            w = n_features // h
            ratio = max(h, w) / min(h, w)  # Aspect ratio
            if ratio < best_ratio:
                best_ratio = ratio
                best_h, best_w = h, w

    return best_h, best_w


def _reshape_data_for_cnn(X_train, X_test, target_shape, data_type):
    """Reshape data appropriately for CNN"""
    try:
        if data_type == "tabular_data":
            # Reshape 2D tabular data to image-like format
            n_samples_train = X_train.shape[0]
            n_samples_test = X_test.shape[0]

            # Target shape is (n_samples, height, width, channels)
            target_h, target_w, target_c = (
                target_shape[1],
                target_shape[2],
                target_shape[3],
            )

            # Pad with zeros if needed
            n_features = X_train.shape[1]
            needed_features = target_h * target_w

            if n_features < needed_features:
                # Pad with zeros
                pad_width = ((0, 0), (0, needed_features - n_features))
                X_train_padded = np.pad(
                    X_train, pad_width, mode="constant", constant_values=0
                )
                X_test_padded = np.pad(
                    X_test, pad_width, mode="constant", constant_values=0
                )
            else:
                # Truncate if needed
                X_train_padded = X_train[:, :needed_features]
                X_test_padded = X_test[:, :needed_features]

            # Reshape to image format
            X_train_reshaped = X_train_padded.reshape(
                n_samples_train, target_h, target_w, target_c
            )
            X_test_reshaped = X_test_padded.reshape(
                n_samples_test, target_h, target_w, target_c
            )

            print(
                f"✅ Reshaped tabular data: {X_train.shape} → {X_train_reshaped.shape}"
            )

        elif data_type == "grayscale_images":
            # Add channel dimension to 3D grayscale data
            X_train_reshaped = X_train.reshape(*X_train.shape, 1)
            X_test_reshaped = X_test.reshape(*X_test.shape, 1)

            print(
                f"✅ Added channel dimension: {X_train.shape} → {X_train_reshaped.shape}"
            )

        else:  # color_images
            # Already in correct format
            X_train_reshaped = X_train
            X_test_reshaped = X_test

            print(f"✅ Data already in correct format: {X_train.shape}")

        # Normalize to [0, 1] range
        X_train_reshaped = X_train_reshaped.astype("float32")
        X_test_reshaped = X_test_reshaped.astype("float32")

        # Check if data needs normalization
        if X_train_reshaped.max() > 1.0:
            print("📊 Normalizing data to [0, 1] range...")
            X_train_reshaped = X_train_reshaped / X_train_reshaped.max()
            X_test_reshaped = X_test_reshaped / X_train_reshaped.max()

        return X_train_reshaped, X_test_reshaped, True

    except Exception as e:
        print(f"❌ Error reshaping data: {e}")
        return X_train, X_test, False


def _safe_evaluate_cnn_model(X_train, y_train, X_test, y_true, **kwargs):
    """Safely evaluate a CNN model configuration with timeout protection"""
    global _consecutive_failures

    try:
        start_time = time.time()

        if not HAS_TENSORFLOW:
            print("❌ TensorFlow not available for CNN evaluation")
            return 0.0, 0.0, False

        # Get model architecture parameters
        architecture = kwargs.get("architecture", "simple")
        optimizer = kwargs.get("optimizer", "adam")
        learning_rate = kwargs.get("learning_rate", 0.001)
        epochs = kwargs.get("epochs", 10)
        batch_size = kwargs.get("batch_size", 32)

        # Build model
        model = _build_cnn_model(
            X_train.shape[1:], len(np.unique(y_train)), architecture
        )

        # Compile model
        if optimizer == "adam":
            opt = Adam(learning_rate=learning_rate)
        else:
            opt = SGD(learning_rate=learning_rate)

        model.compile(
            optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
        )

        # Train model with timeout protection
        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=0,
        )

        # Check if training took too long
        training_time = time.time() - start_time
        if training_time > _max_time_per_model:
            print(f"⏰ CNN timeout after {training_time:.1f}s")
            return 0.0, 0.0, False

        # Make predictions
        y_pred = model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)

        accuracy = accuracy_score(y_true, y_pred_classes)
        f1 = f1_score(y_true, y_pred_classes, average="weighted", zero_division=0)

        # Track consecutive failures
        if accuracy < _min_accuracy_threshold:
            _consecutive_failures += 1
        else:
            _consecutive_failures = 0

        # Clean up memory
        del model
        tf.keras.backend.clear_session()

        return accuracy, f1, True

    except Exception as e:
        print(f"❌ CNN evaluation failed: {e}")
        _consecutive_failures += 1
        return 0.0, 0.0, False


def _build_cnn_model(input_shape, n_classes, architecture="simple"):
    """Build CNN model with different architectures"""

    model = Sequential()

    if architecture == "simple":
        # Simple CNN for small images or tabular data
        model.add(
            Conv2D(
                32, (3, 3), activation="relu", input_shape=input_shape, padding="same"
            )
        )
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(n_classes, activation="softmax"))

    elif architecture == "medium":
        # Medium CNN for moderate sized images
        model.add(
            Conv2D(
                32, (3, 3), activation="relu", input_shape=input_shape, padding="same"
            )
        )
        model.add(Conv2D(32, (3, 3), activation="relu", padding="same"))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
        model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(n_classes, activation="softmax"))

    elif architecture == "deep":
        # Deeper CNN for larger images
        model.add(
            Conv2D(
                32, (3, 3), activation="relu", input_shape=input_shape, padding="same"
            )
        )
        model.add(Conv2D(32, (3, 3), activation="relu", padding="same"))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
        model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
        model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(256, (3, 3), activation="relu", padding="same"))
        model.add(Flatten())
        model.add(Dense(256, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(n_classes, activation="softmax"))

    return model


def _get_smart_architectures(image_height, image_width, n_samples):
    """Get smart CNN architectures based on image characteristics"""

    architectures = []

    # Determine suitable architectures based on image size
    min_dim = min(image_height, image_width)

    if min_dim < 16:
        # Very small images - only simple architecture
        architectures = ["simple"]
    elif min_dim < 32:
        # Small images - simple and medium
        architectures = ["simple", "medium"]
    else:
        # Larger images - can handle all architectures
        if n_samples > 1000:
            architectures = ["simple", "medium", "deep"]
        else:
            architectures = ["simple", "medium"]

    return architectures


def _optimize_architecture_and_optimizer(X_train, y_train, X_test, y_true, opts):
    """Optimize architecture and optimizer together for efficiency"""
    global _consecutive_failures
    start_time = time.time()
    _consecutive_failures = 0

    max_acc = -np.inf
    best_f1 = 0.0

    # Get image dimensions
    image_height, image_width = X_train.shape[1], X_train.shape[2]
    n_samples = X_train.shape[0]

    # Get smart architectures
    architectures = _get_smart_architectures(image_height, image_width, n_samples)

    # Smart optimizer selection
    optimizers = ["adam", "sgd"]

    # Create architecture-optimizer combinations
    configs = []
    for arch in architectures:
        for opt in optimizers:
            configs.append({"architecture": arch, "optimizer": opt})

    best_config = configs[0]

    with tqdm(configs, desc="Optimizing Architecture & Optimizer", leave=False) as pbar:
        for config in pbar:
            # Early stopping conditions
            if _consecutive_failures >= _max_consecutive_failures:
                pbar.set_description("Architecture & Optimizer (POOR ACCURACY)")
                break

            test_opts = opts.copy()
            test_opts.update(config)

            accuracy, f1, success = _safe_evaluate_cnn_model(
                X_train, y_train, X_test, y_true, **test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_config = config

            pbar.set_postfix(
                {
                    "arch": config["architecture"][:6],
                    "opt": config["optimizer"][:4],
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts.update(best_config)
    return opts, max_acc, best_f1


def _optimize_learning_rate(X_train, y_train, X_test, y_true, opts):
    """Optimize learning rate with early stopping"""
    global _consecutive_failures
    start_time = time.time()
    _consecutive_failures = 0

    max_acc = -np.inf
    best_f1 = 0.0

    # Smart learning rate range based on optimizer
    if opts["optimizer"] == "adam":
        variable_array = [0.001, 0.0001, 0.01, 0.0005]  # Adam rates
    else:  # SGD
        variable_array = [0.1, 0.01, 0.001, 0.05]  # SGD rates

    best_val = variable_array[0]  # Default

    with tqdm(variable_array, desc="Optimizing Learning Rate", leave=False) as pbar:
        for v in pbar:
            if _consecutive_failures >= _max_consecutive_failures:
                pbar.set_description("Learning Rate (POOR ACCURACY)")
                break

            test_opts = opts.copy()
            test_opts["learning_rate"] = v

            accuracy, f1, success = _safe_evaluate_cnn_model(
                X_train, y_train, X_test, y_true, **test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_val = v

            pbar.set_postfix(
                {
                    "lr": f"{v:.4f}",
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts["learning_rate"] = best_val
    return opts, max_acc, best_f1


def _optimize_training_params(X_train, y_train, X_test, y_true, opts):
    """Optimize training parameters together"""
    global _consecutive_failures
    start_time = time.time()
    _consecutive_failures = 0

    max_acc = -np.inf
    best_f1 = 0.0
    n_samples = X_train.shape[0]

    # Smart parameter combinations based on dataset size
    if n_samples < 1000:
        configs = [
            {"epochs": 20, "batch_size": 16},
            {"epochs": 30, "batch_size": 32},
            {"epochs": 15, "batch_size": 8},
        ]
    elif n_samples < 5000:
        configs = [
            {"epochs": 15, "batch_size": 32},
            {"epochs": 20, "batch_size": 64},
            {"epochs": 10, "batch_size": 16},
        ]
    else:
        configs = [
            {"epochs": 10, "batch_size": 64},
            {"epochs": 15, "batch_size": 128},
            {"epochs": 8, "batch_size": 32},
        ]

    best_config = configs[0]

    with tqdm(configs, desc="Optimizing Training Params", leave=False) as pbar:
        for config in pbar:
            # Early stopping conditions
            if _consecutive_failures >= _max_consecutive_failures:
                pbar.set_description("Training Params (POOR ACCURACY)")
                break

            test_opts = opts.copy()
            test_opts.update(config)

            accuracy, f1, success = _safe_evaluate_cnn_model(
                X_train, y_train, X_test, y_true, **test_opts
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_config = config

            pbar.set_postfix(
                {
                    "epochs": config["epochs"],
                    "batch": config["batch_size"],
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts.update(best_config)
    return opts, max_acc, best_f1


def _optimize_cnn(X_train, y_train, X_test, y_true, cycles=2):
    """
    ROBUST CNN optimization that handles both tabular and image data.

    Always returns exactly 5 values: (opts, accuracy, f1, accuracy_history, f1_history)
    """

    print("🧠 ROBUST CNN Optimization")
    print("Focus: Reliability + Data Handling")
    print("=" * 60)

    try:
        # Critical suitability check
        suitable, data_type, target_shape = _check_cnn_suitability(X_train, y_train)

        if not suitable:
            print(f"\n❌ Dataset unsuitable for CNN optimization! Reason: {data_type}")
            print("Returning safe default values.")
            print("Recommendation: Use MLP/Neural Network for tabular data")
            return _create_default_return_values()

        # Reshape data appropriately
        print(f"\n🔄 Preparing data for CNN...")
        X_train_reshaped, X_test_reshaped, reshape_success = _reshape_data_for_cnn(
            X_train, X_test, target_shape, data_type
        )

        if not reshape_success:
            print("❌ Data reshaping failed!")
            return _create_default_return_values()

        n_samples, height, width, channels = X_train_reshaped.shape
        n_classes = len(np.unique(y_train))

        print(f"Final data shape: {X_train_reshaped.shape}")
        print(f"Image size: {height}x{width}x{channels}, Classes: {n_classes}")

        # CNN warnings based on final shape
        if height < 16 or width < 16:
            print("⚠️  WARNING: Very small images for CNN!")
            print("CNNs work better with larger images (32x32+)")

        if n_samples < 500:
            print("⚠️  WARNING: Small dataset for CNN!")
            print("CNNs typically need thousands of samples.")

        # Quick baseline check
        print("\nRunning baseline CNN check...")
        baseline_start = time.time()

        try:
            baseline_opts = {
                "architecture": "simple",
                "optimizer": "adam",
                "learning_rate": 0.001,
                "epochs": 5,  # Quick baseline
                "batch_size": 32,
            }

            baseline_acc, baseline_f1, baseline_success = _safe_evaluate_cnn_model(
                X_train_reshaped, y_train, X_test_reshaped, y_true, **baseline_opts
            )
            baseline_time = time.time() - baseline_start

            if baseline_success:
                print(
                    f"Baseline CNN: {baseline_acc:.4f} (trained in {baseline_time:.1f}s)"
                )
            else:
                print("⚠️  Baseline CNN failed - proceeding with caution")
                baseline_acc = 0.0

        except Exception as e:
            print(f"Baseline check failed: {e}")
            baseline_acc = 0.0

        # Define initial parameters with smart defaults
        opts = {
            "architecture": "simple",
            "optimizer": "adam",
            "learning_rate": 0.001,
            "epochs": 10,
            "batch_size": 32,
        }

        # Track results with guaranteed initialization
        ma_vec = []
        f1_vec = []
        final_accuracy = 0.0
        final_f1 = 0.0

        # Main optimization loop with error handling
        with tqdm(
            range(cycles), desc="Robust CNN Optimization", position=0
        ) as cycle_pbar:
            for c in cycle_pbar:
                try:
                    cycle_start_time = time.time()
                    cycle_pbar.set_description(f"CNN Cycle {c + 1}/{cycles}")

                    # Core optimizations in order of importance
                    try:
                        opts, _, _ = _optimize_architecture_and_optimizer(
                            X_train_reshaped, y_train, X_test_reshaped, y_true, opts
                        )
                    except Exception as e:
                        print(f"❌ Architecture optimization failed: {e}")

                    try:
                        opts, _, _ = _optimize_learning_rate(
                            X_train_reshaped, y_train, X_test_reshaped, y_true, opts
                        )
                    except Exception as e:
                        print(f"❌ Learning rate optimization failed: {e}")

                    try:
                        opts, ma, f1 = _optimize_training_params(
                            X_train_reshaped, y_train, X_test_reshaped, y_true, opts
                        )

                        final_accuracy = ma
                        final_f1 = f1

                    except Exception as e:
                        print(f"❌ Training parameter optimization failed: {e}")
                        ma = f1 = 0.0

                    # Record results
                    ma_vec.append(ma)
                    f1_vec.append(f1)

                    cycle_time = time.time() - cycle_start_time

                    cycle_pbar.set_postfix(
                        {
                            "accuracy": f"{ma:.4f}",
                            "f1": f"{f1:.4f}",
                            "best_overall": f"{max(ma_vec) if ma_vec else 0:.4f}",
                            "cycle_time": f"{cycle_time:.1f}s",
                            "arch": opts["architecture"][:6],
                            "opt": opts["optimizer"][:4],
                            "lr": f'{opts["learning_rate"]:.4f}',
                            "epochs": opts["epochs"],
                            "baseline_beat": f"{ma - baseline_acc:+.4f}",
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

        print(f"\n🎯 Robust CNN optimization completed!")
        print(f"Final accuracy: {final_accuracy:.4f}")
        print(f"Final F1: {final_f1:.4f}")

        # GUARANTEE: Always return exactly 5 values
        return opts, final_accuracy, final_f1, ma_vec, f1_vec

    except Exception as e:
        print(f"❌ CRITICAL ERROR in CNN optimization: {e}")
        print("Returning safe default values to prevent crash.")
        return _create_default_return_values()
