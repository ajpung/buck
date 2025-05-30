import warnings
import numpy as np
import time
from tqdm.auto import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import gc

# Suppress TensorFlow warnings for cleaner progress bars
tf.get_logger().setLevel("ERROR")
warnings.filterwarnings("ignore")

# Global efficiency controls for CNN (very aggressive due to training cost)
_max_time_per_step = 1800  # 30 minutes max per optimization step
_max_time_per_model = 600  # 10 minutes max per model training
_min_accuracy_threshold = 0.15  # Stop if accuracy is consistently terrible
_consecutive_failures = 0
_max_consecutive_failures = 2  # More aggressive for CNNs

# Hardware detection
_has_gpu = len(tf.config.list_physical_devices("GPU")) > 0


def _check_hardware_suitability(X_train):
    """Check if hardware is suitable for CNN training"""
    n_samples = X_train.shape[0]
    image_size = X_train.shape[1] * X_train.shape[2] if len(X_train.shape) > 2 else 0

    print(f"🔍 Hardware Check:")
    print(
        f"  GPU Available: {'✅ Yes' if _has_gpu else '❌ No (CPU only - will be VERY slow!)'}"
    )
    print(f"  Dataset Size: {n_samples} images")
    print(
        f"  Image Dimensions: {X_train.shape[1:] if len(X_train.shape) > 1 else 'Unknown'}"
    )

    if not _has_gpu:
        print("⚠️  WARNING: No GPU detected!")
        print("CNN training on CPU is extremely slow. Consider:")
        print("  - Using Google Colab (free GPU)")
        print("  - Using cloud services (AWS, GCP)")
        print("  - Using traditional ML algorithms instead")
        if n_samples > 1000:
            print("  - Reducing dataset size for CPU training")

    if n_samples < 100:
        print("⚠️  WARNING: Very small dataset for CNNs!")
        print("CNNs typically need thousands of images. Consider:")
        print("  - Data augmentation")
        print("  - Transfer learning")
        print("  - Traditional ML algorithms")

    return _has_gpu


def _safe_evaluate_cnn(
    X_train, y_train, X_test, y_true, architecture, compile_opts, fit_opts
):
    """Safely evaluate a CNN configuration with timeout protection"""
    global _consecutive_failures

    try:
        # Clear session to prevent memory issues
        keras.backend.clear_session()
        start_time = time.time()

        # Build model
        model = _build_cnn_model(
            X_train.shape[1:], len(np.unique(y_train)), architecture
        )
        model.compile(**compile_opts)

        # Add timeout callback
        class TimeoutCallback(keras.callbacks.Callback):
            def __init__(self, max_time):
                self.max_time = max_time
                self.start_time = time.time()

            def on_epoch_end(self, epoch, logs=None):
                if time.time() - self.start_time > self.max_time:
                    print(f"⏰ Training timeout after {self.max_time}s")
                    self.model.stop_training = True

        # Prepare callbacks
        callbacks = fit_opts.get("callbacks", []).copy()
        callbacks.append(TimeoutCallback(_max_time_per_model))

        # Train model with reduced verbosity and timeout
        history = model.fit(
            X_train,
            y_train,
            validation_split=0.2,
            verbose=0,
            callbacks=callbacks,
            **{k: v for k, v in fit_opts.items() if k != "callbacks"},
        )

        # Check if training took too long
        training_time = time.time() - start_time
        if training_time > _max_time_per_model:
            print(f"⏰ CNN timeout after {training_time:.1f}s")
            del model
            gc.collect()
            return 0.0, 0.0, False, 0.0

        # Evaluate
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)

        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        val_acc = (
            history.history.get("val_accuracy", [0])[-1]
            if "val_accuracy" in history.history
            else 0
        )

        # Track consecutive failures
        if accuracy < _min_accuracy_threshold:
            _consecutive_failures += 1
        else:
            _consecutive_failures = 0

        # Clean up
        del model
        gc.collect()

        return accuracy, f1, True, val_acc

    except Exception as e:
        print(f"CNN evaluation failed: {e}")
        keras.backend.clear_session()
        gc.collect()
        _consecutive_failures += 1
        return 0.0, 0.0, False, 0.0


def _build_cnn_model(input_shape, num_classes, architecture):
    """Build CNN model based on architecture configuration"""
    model = keras.Sequential()

    # Input layer
    model.add(layers.Input(shape=input_shape))

    # Convolutional blocks
    for i, block in enumerate(architecture["conv_blocks"]):
        # Convolutional layer
        model.add(
            layers.Conv2D(
                filters=block["filters"],
                kernel_size=block["kernel_size"],
                activation=architecture["activation"],
                padding="same",
            )
        )

        # Batch normalization (if enabled)
        if architecture.get("batch_norm", False):
            model.add(layers.BatchNormalization())

        # Pooling layer
        if block.get("pool_size"):
            model.add(layers.MaxPooling2D(pool_size=block["pool_size"]))

        # Dropout (if specified)
        if block.get("dropout", 0) > 0:
            model.add(layers.Dropout(block["dropout"]))

    # Global pooling or flatten
    if architecture.get("global_pooling", False):
        model.add(layers.GlobalAveragePooling2D())
    else:
        model.add(layers.Flatten())

    # Dense layers
    for dense_units in architecture["dense_layers"]:
        model.add(layers.Dense(dense_units, activation=architecture["activation"]))
        if architecture.get("dense_dropout", 0) > 0:
            model.add(layers.Dropout(architecture["dense_dropout"]))

    # Output layer
    if num_classes == 2:
        model.add(layers.Dense(1, activation="sigmoid"))
    else:
        model.add(layers.Dense(num_classes, activation="softmax"))

    return model


def _get_fast_cnn_architectures(input_shape, n_samples):
    """Get fast CNN architectures based on dataset size and image dimensions"""

    height, width = input_shape[:2]

    # Very lightweight architectures for speed
    architectures = []

    if n_samples < 500:
        # Tiny dataset - very simple architectures
        architectures = [
            {
                "name": "Tiny-1",
                "conv_blocks": [
                    {"filters": 16, "kernel_size": (3, 3), "pool_size": (2, 2)},
                    {"filters": 32, "kernel_size": (3, 3), "pool_size": (2, 2)},
                ],
                "dense_layers": [64],
                "activation": "relu",
                "batch_norm": False,
                "global_pooling": True,  # Reduces parameters
                "dense_dropout": 0.3,
            },
            {
                "name": "Tiny-2",
                "conv_blocks": [
                    {"filters": 32, "kernel_size": (3, 3), "pool_size": (2, 2)},
                ],
                "dense_layers": [32],
                "activation": "relu",
                "batch_norm": False,
                "global_pooling": True,
                "dense_dropout": 0.5,
            },
        ]
    elif n_samples < 2000:
        # Small dataset - simple architectures
        architectures = [
            {
                "name": "Small-1",
                "conv_blocks": [
                    {"filters": 32, "kernel_size": (3, 3), "pool_size": (2, 2)},
                    {"filters": 64, "kernel_size": (3, 3), "pool_size": (2, 2)},
                ],
                "dense_layers": [128],
                "activation": "relu",
                "batch_norm": False,
                "global_pooling": False,
                "dense_dropout": 0.5,
            },
            {
                "name": "Small-BN",
                "conv_blocks": [
                    {"filters": 32, "kernel_size": (3, 3), "pool_size": (2, 2)},
                    {"filters": 64, "kernel_size": (3, 3), "pool_size": (2, 2)},
                ],
                "dense_layers": [128],
                "activation": "relu",
                "batch_norm": True,
                "global_pooling": True,
                "dense_dropout": 0.3,
            },
        ]
    else:
        # Larger dataset - can handle slightly more complex architectures
        architectures = [
            {
                "name": "Medium-1",
                "conv_blocks": [
                    {"filters": 32, "kernel_size": (3, 3), "pool_size": (2, 2)},
                    {"filters": 64, "kernel_size": (3, 3), "pool_size": (2, 2)},
                    {"filters": 128, "kernel_size": (3, 3), "pool_size": (2, 2)},
                ],
                "dense_layers": [256],
                "activation": "relu",
                "batch_norm": True,
                "global_pooling": False,
                "dense_dropout": 0.5,
            },
            {
                "name": "Medium-GlobalPool",
                "conv_blocks": [
                    {"filters": 64, "kernel_size": (3, 3), "pool_size": (2, 2)},
                    {"filters": 128, "kernel_size": (3, 3), "pool_size": (2, 2)},
                    {"filters": 256, "kernel_size": (3, 3)},
                ],
                "dense_layers": [128],
                "activation": "relu",
                "batch_norm": True,
                "global_pooling": True,
                "dense_dropout": 0.3,
            },
        ]

    return architectures


def _optimize_architecture_and_optimizer(X_train, y_train, X_test, y_true, opts):
    """Optimize architecture and optimizer together for maximum efficiency"""
    global _consecutive_failures
    start_time = time.time()
    _consecutive_failures = 0

    max_acc = -np.inf
    best_f1 = 0.0
    n_samples = X_train.shape[0]

    # Get fast architectures
    architectures = _get_fast_cnn_architectures(X_train.shape[1:], n_samples)

    # Smart optimizer selection (fewer options)
    if not _has_gpu:
        optimizers = [
            {"name": "adam", "lr": 0.001},  # Only one for CPU
        ]
    else:
        optimizers = [
            {"name": "adam", "lr": 0.001},
            {"name": "adam", "lr": 0.0001},
            {"name": "rmsprop", "lr": 0.001},
        ]

    # Create architecture-optimizer combinations
    configs = []
    for arch in architectures:
        for opt in optimizers:
            configs.append({"architecture": arch, "optimizer": opt})

    best_config = configs[0]

    with tqdm(
        configs, desc="Optimizing CNN Architecture & Optimizer", leave=False
    ) as pbar:
        for config in pbar:
            # Early stopping conditions
            if time.time() - start_time > _max_time_per_step:
                pbar.set_description("CNN Architecture (TIME LIMIT)")
                break
            if _consecutive_failures >= _max_consecutive_failures:
                pbar.set_description("CNN Architecture (POOR ACCURACY)")
                break

            # Setup optimizer
            test_opts = opts.copy()
            if config["optimizer"]["name"] == "adam":
                test_opts["compile"]["optimizer"] = keras.optimizers.Adam(
                    learning_rate=config["optimizer"]["lr"]
                )
            elif config["optimizer"]["name"] == "rmsprop":
                test_opts["compile"]["optimizer"] = keras.optimizers.RMSprop(
                    learning_rate=config["optimizer"]["lr"]
                )

            accuracy, f1, success, val_acc = _safe_evaluate_cnn(
                X_train,
                y_train,
                X_test,
                y_true,
                config["architecture"],
                test_opts["compile"],
                test_opts["fit"],
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_config = config

            arch_name = config["architecture"]["name"]
            opt_info = f"{config['optimizer']['name']}({config['optimizer']['lr']:.0e})"

            pbar.set_postfix(
                {
                    "arch": arch_name[:8],
                    "opt": opt_info[:10],
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "val": f"{val_acc:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    # Set best configuration
    opts["architecture"] = best_config["architecture"]
    if best_config["optimizer"]["name"] == "adam":
        opts["compile"]["optimizer"] = keras.optimizers.Adam(
            learning_rate=best_config["optimizer"]["lr"]
        )
    elif best_config["optimizer"]["name"] == "rmsprop":
        opts["compile"]["optimizer"] = keras.optimizers.RMSprop(
            learning_rate=best_config["optimizer"]["lr"]
        )

    return opts, max_acc, best_f1


def _optimize_training_params(X_train, y_train, X_test, y_true, opts):
    """Optimize training parameters with focus on speed"""
    global _consecutive_failures
    start_time = time.time()
    _consecutive_failures = 0

    max_acc = -np.inf
    best_f1 = 0.0
    n_samples = X_train.shape[0]

    # Fast training configurations
    if not _has_gpu:
        # CPU configurations - very fast training
        configs = [
            {"epochs": 10, "batch_size": 32, "early_stopping": True, "patience": 3},
            {"epochs": 20, "batch_size": 64, "early_stopping": True, "patience": 5},
        ]
    elif n_samples < 500:
        # Small dataset configs
        configs = [
            {"epochs": 20, "batch_size": 16, "early_stopping": True, "patience": 5},
            {"epochs": 30, "batch_size": 32, "early_stopping": True, "patience": 7},
        ]
    elif n_samples < 2000:
        # Medium dataset configs
        configs = [
            {"epochs": 30, "batch_size": 32, "early_stopping": True, "patience": 7},
            {"epochs": 50, "batch_size": 64, "early_stopping": True, "patience": 10},
        ]
    else:
        # Larger dataset configs
        configs = [
            {"epochs": 50, "batch_size": 64, "early_stopping": True, "patience": 10},
            {"epochs": 100, "batch_size": 128, "early_stopping": True, "patience": 15},
        ]

    best_config = configs[0]

    with tqdm(configs, desc="Optimizing Training Params", leave=False) as pbar:
        for config in pbar:
            # Early stopping conditions
            if time.time() - start_time > _max_time_per_step:
                pbar.set_description("Training Params (TIME LIMIT)")
                break
            if _consecutive_failures >= _max_consecutive_failures:
                pbar.set_description("Training Params (POOR ACCURACY)")
                break

            test_opts = opts.copy()
            test_opts["fit"]["epochs"] = config["epochs"]
            test_opts["fit"]["batch_size"] = config["batch_size"]

            # Add early stopping
            callbacks = []
            if config.get("early_stopping", False):
                callbacks.append(
                    keras.callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=config["patience"],
                        restore_best_weights=True,
                    )
                )
            test_opts["fit"]["callbacks"] = callbacks

            accuracy, f1, success, val_acc = _safe_evaluate_cnn(
                X_train,
                y_train,
                X_test,
                y_true,
                opts["architecture"],
                test_opts["compile"],
                test_opts["fit"],
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_config = config

            pbar.set_postfix(
                {
                    "epochs": config["epochs"],
                    "batch": config["batch_size"],
                    "early": "Y" if config.get("early_stopping") else "N",
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    # Set best configuration
    opts["fit"]["epochs"] = best_config["epochs"]
    opts["fit"]["batch_size"] = best_config["batch_size"]

    callbacks = []
    if best_config.get("early_stopping", False):
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=best_config["patience"],
                restore_best_weights=True,
            )
        )
    opts["fit"]["callbacks"] = callbacks

    return opts, max_acc, best_f1


def _optimize_cnn(X_train, y_train, X_test, y_true, cycles=1):
    """
    FAST CNN hyperparameter optimization for image classification.

    :param X_train: Training images (N, H, W, C) or (N, H, W)
    :param y_train: Training labels
    :param X_test: Test images
    :param y_true: True labels for test data
    :param cycles: Number of optimization cycles (keep low for CNNs!)
    :return: optimized options, best accuracy, best f1, accuracy history, f1 history
    """

    print("🧠 FAST CNN Optimization - Hardware & Dataset Analysis")
    print("=" * 60)

    # Hardware check
    has_gpu = _check_hardware_suitability(X_train)

    # Ensure proper data format
    if len(X_train.shape) == 3:  # Add channel dimension if grayscale
        X_train = np.expand_dims(X_train, axis=-1)
        X_test = np.expand_dims(X_test, axis=-1)

    # Normalize data
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    n_samples, height, width = X_train.shape[:3]

    # More dataset warnings
    if height < 32 or width < 32:
        print("⚠️  WARNING: Very small images - CNNs may not be optimal")
    if height > 224 or width > 224:
        print("⚠️  WARNING: Large images will be slow to train")
        print("Consider resizing images to 224x224 or smaller")

    # Encode labels if needed
    if len(y_train.shape) == 1:
        num_classes = len(np.unique(y_train))
        if num_classes > 2:
            y_train = keras.utils.to_categorical(y_train, num_classes)
            y_true_encoded = keras.utils.to_categorical(y_true, num_classes)
        else:
            y_true_encoded = y_true
    else:
        y_true_encoded = y_true
        num_classes = y_train.shape[1]

    print(f"Dataset: {n_samples} images, {height}x{width}, {num_classes} classes")

    # Quick baseline check with minimal CNN
    print("Running baseline check...")
    baseline_start = time.time()

    try:
        baseline_model = keras.Sequential(
            [
                layers.Input(shape=X_train.shape[1:]),
                layers.Conv2D(16, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Flatten(),
                layers.Dense(32, activation="relu"),
                layers.Dense(
                    num_classes if num_classes > 2 else 1,
                    activation="softmax" if num_classes > 2 else "sigmoid",
                ),
            ]
        )

        baseline_model.compile(
            optimizer="adam",
            loss=(
                "categorical_crossentropy" if num_classes > 2 else "binary_crossentropy"
            ),
            metrics=["accuracy"],
        )

        baseline_model.fit(
            X_train, y_train, epochs=3, batch_size=32, verbose=0, validation_split=0.2
        )
        baseline_acc = baseline_model.evaluate(X_test, y_true_encoded, verbose=0)[1]
        baseline_time = time.time() - baseline_start

        print(f"Baseline CNN: {baseline_acc:.4f} (trained in {baseline_time:.1f}s)")

        if baseline_time > 60:
            print("⚠️  Slow baseline - CNN optimization will take significant time")

        del baseline_model
        keras.backend.clear_session()

    except Exception as e:
        print(f"Baseline check failed: {e}")
        baseline_acc = 0.0

    # Define initial options with smart defaults
    opts = {
        "architecture": {
            "conv_blocks": [
                {"filters": 32, "kernel_size": (3, 3), "pool_size": (2, 2)},
                {"filters": 64, "kernel_size": (3, 3), "pool_size": (2, 2)},
            ],
            "dense_layers": [128],
            "activation": "relu",
            "batch_norm": False,
            "global_pooling": False,
            "dense_dropout": 0.5,
        },
        "compile": {
            "optimizer": keras.optimizers.Adam(learning_rate=0.001),
            "loss": (
                "sparse_categorical_crossentropy"
                if num_classes > 2 and len(y_train.shape) == 1
                else (
                    "categorical_crossentropy"
                    if num_classes > 2
                    else "binary_crossentropy"
                )
            ),
            "metrics": ["accuracy"],
        },
        "fit": {
            "epochs": 20 if not has_gpu else 50,
            "batch_size": 32,
            "callbacks": [
                keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=5, restore_best_weights=True
                )
            ],
        },
    }

    # Track results
    ma_vec = []
    f1_vec = []

    # Main optimization loop (keep cycles low for CNNs!)
    with tqdm(range(cycles), desc="FAST CNN Optimization", position=0) as cycle_pbar:
        for c in cycle_pbar:
            cycle_start_time = time.time()
            cycle_pbar.set_description(f"CNN Cycle {c + 1}/{cycles}")

            # Core optimizations (minimal for speed)
            opts, _, _ = _optimize_architecture_and_optimizer(
                X_train, y_train, X_test, y_true_encoded, opts
            )
            opts, ma, f1 = _optimize_training_params(
                X_train, y_train, X_test, y_true_encoded, opts
            )

            ma_vec.append(ma)
            f1_vec.append(f1)

            cycle_time = time.time() - cycle_start_time

            cycle_pbar.set_postfix(
                {
                    "accuracy": f"{ma:.4f}",
                    "f1": f"{f1:.4f}",
                    "best_overall": f"{max(ma_vec):.4f}",
                    "cycle_time": f"{cycle_time:.1f}s",
                    "arch": opts["architecture"].get("name", "Custom")[:8],
                    "epochs": opts["fit"]["epochs"],
                    "batch": opts["fit"]["batch_size"],
                    "baseline_beat": f"{ma - baseline_acc:+.4f}",
                }
            )

    print(
        f"\n🎯 CNN optimization completed in {time.time() - baseline_start:.1f}s total"
    )

    return opts, ma, f1, ma_vec, f1_vec


def _analyze_cnn_performance(X_train, y_train, X_test, y_true, best_opts):
    """Analyze CNN performance and provide insights"""

    print("\n" + "=" * 70)
    print("FAST CNN PERFORMANCE ANALYSIS")
    print("=" * 70)

    # Encode labels if needed for final evaluation
    if len(y_train.shape) == 1:
        num_classes = len(np.unique(y_train))
        if num_classes > 2:
            y_train_encoded = keras.utils.to_categorical(y_train, num_classes)
            y_true_encoded = keras.utils.to_categorical(y_true, num_classes)
        else:
            y_train_encoded = y_train
            y_true_encoded = y_true
    else:
        y_train_encoded = y_train
        y_true_encoded = y_true

    # Train final model
    print("Training final CNN with best parameters...")
    start_time = time.time()

    final_model = _build_cnn_model(
        X_train.shape[1:], len(np.unique(y_train)), best_opts["architecture"]
    )
    final_model.compile(**best_opts["compile"])

    history = final_model.fit(
        X_train, y_train_encoded, validation_split=0.2, verbose=0, **best_opts["fit"]
    )

    training_time = time.time() - start_time

    # Evaluate
    y_pred_proba = final_model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")

    print(f"\n🎯 Final CNN Performance:")
    print(f"  Test Accuracy: {accuracy:.4f}")
    print(f"  Test F1 Score: {f1:.4f}")
    print(f"  Training Time: {training_time:.1f} seconds")

    # Architecture analysis
    print(f"\n🏗️  Optimal Architecture:")
    arch = best_opts["architecture"]
    print(f"  Name: {arch.get('name', 'Custom')}")
    print(f"  Conv Blocks: {len(arch['conv_blocks'])}")

    total_filters = sum(block["filters"] for block in arch["conv_blocks"])
    print(f"  Total Filters: {total_filters}")
    print(f"  Dense Layers: {arch['dense_layers']}")
    print(f"  Activation: {arch['activation']}")
    print(f"  Batch Normalization: {arch.get('batch_norm', False)}")
    print(f"  Global Pooling: {arch.get('global_pooling', False)}")

    # Training analysis
    print(f"\n📈 Training Configuration:")
    print(f"  Optimizer: {str(best_opts['compile']['optimizer'])[:30]}")
    print(f"  Epochs: {best_opts['fit']['epochs']}")
    print(f"  Batch Size: {best_opts['fit']['batch_size']}")
    print(f"  Early Stopping: {'Yes' if best_opts['fit'].get('callbacks') else 'No'}")

    # Model complexity
    total_params = final_model.count_params()
    trainable_params = int(
        np.sum([keras.backend.count_params(p) for p in final_model.trainable_weights])
    )

    print(f"\n🧮 Model Complexity:")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")

    # Training history analysis
    if len(history.history) > 0:
        final_loss = history.history.get("loss", [0])[-1]
        final_val_loss = history.history.get("val_loss", [0])[-1]
        final_val_acc = history.history.get("val_accuracy", [0])[-1]

        print(f"\n📊 Training History:")
        print(f"  Final Training Loss: {final_loss:.4f}")
        print(f"  Final Validation Loss: {final_val_loss:.4f}")
        print(f"  Final Validation Accuracy: {final_val_acc:.4f}")
        print(f"  Epochs Trained: {len(history.history.get('loss', []))}")

    # Performance insights
    n_samples = X_train.shape[0]
    print(f"\n💡 Performance Insights:")

    if accuracy > 0.9:
        print(f"  ✅ Excellent performance - CNN is working well")
    elif accuracy > 0.7:
        print(f"  ✅ Good performance - room for improvement with more training")
    elif accuracy > 0.5:
        print(
            f"  ⚠️  Moderate performance - consider data augmentation or architecture changes"
        )
    else:
        print(
            f"  ❌ Poor performance - check data quality, preprocessing, or try different approach"
        )

    if training_time > 300:  # 5 minutes
        print(
            f"  ⏰ Long training time - consider smaller architecture or fewer epochs"
        )

    if total_params > n_samples * 10:
        print(f"  ⚠️  Many parameters vs samples - risk of overfitting")

    # Recommendations
    print(f"\n🚀 Recommendations:")
    if accuracy < 0.8 and n_samples < 1000:
        print(f"  💡 Small dataset - consider transfer learning or data augmentation")
    if not _has_gpu and training_time > 60:
        print(f"  💡 Consider using GPU for faster training")
    if total_params > 100000:
        print(f"  💡 Large model - consider model compression for deployment")

    # Cleanup
    del final_model
    keras.backend.clear_session()
    gc.collect()

    return {
        "accuracy": accuracy,
        "f1": f1,
        "training_time": training_time,
        "total_params": total_params,
        "architecture_name": arch.get("name", "Custom"),
        "epochs_trained": len(history.history.get("loss", [])),
    }


# Example usage function
def example_usage():
    """Example of FAST CNN optimization"""
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    import numpy as np

    print("🚀 FAST CNN Optimization")
    print("Focus: Hardware-Aware + Speed")
    print("=" * 50)

    # Generate sample image data (simulated)
    print("Generating sample image data...")

    # Create fake image data for demo
    n_samples = 1000
    height, width, channels = 32, 32, 3

    X = np.random.rand(n_samples, height, width, channels) * 255
    y = np.random.randint(0, 3, n_samples)  # 3 classes

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Dataset: {X_train.shape[0]} images of size {height}x{width}x{channels}")
    print("⚠️  Note: Using random data for demo - real images would work better!")

    # Run FAST CNN optimization
    best_opts, best_acc, best_f1, acc_history, f1_history = _optimize_cnn(
        X_train, y_train, X_test, y_test, cycles=1  # Keep cycles=1 for demo
    )

    print(f"\n🎯 OPTIMIZATION COMPLETE!")
    print(f"Best CNN Accuracy: {best_acc:.4f}")
    print(f"Best CNN F1: {best_f1:.4f}")

    # Analyze performance
    analysis = _analyze_cnn_performance(X_train, y_train, X_test, y_test, best_opts)

    return best_opts, best_acc, best_f1, acc_history, f1_history
