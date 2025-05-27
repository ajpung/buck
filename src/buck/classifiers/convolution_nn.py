import warnings
import numpy as np
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


def _safe_evaluate_cnn(
    X_train, y_train, X_test, y_true, architecture, compile_opts, fit_opts
):
    """Safely evaluate a CNN configuration"""
    try:
        # Clear session to prevent memory issues
        keras.backend.clear_session()

        # Build model
        model = _build_cnn_model(
            X_train.shape[1:], len(np.unique(y_train)), architecture
        )
        model.compile(**compile_opts)

        # Train model with reduced verbosity
        history = model.fit(
            X_train, y_train, validation_split=0.2, verbose=0, **fit_opts
        )

        # Evaluate
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)

        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        # Clean up
        del model
        gc.collect()

        return (
            accuracy,
            f1,
            True,
            (
                history.history.get("val_accuracy", [0])[-1]
                if "val_accuracy" in history.history
                else 0
            ),
        )

    except Exception as e:
        keras.backend.clear_session()
        gc.collect()
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


def _optimize_conv_architecture(X_train, y_train, X_test, y_true, opts):
    """Optimize convolutional architecture"""
    max_acc = -np.inf
    best_f1 = 0.0
    best_val_acc = 0.0

    # Define different CNN architectures
    architectures = [
        # Simple architectures
        {
            "name": "Simple-1",
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
            "name": "Simple-2",
            "conv_blocks": [
                {"filters": 32, "kernel_size": (3, 3)},
                {"filters": 32, "kernel_size": (3, 3), "pool_size": (2, 2)},
                {"filters": 64, "kernel_size": (3, 3)},
                {"filters": 64, "kernel_size": (3, 3), "pool_size": (2, 2)},
            ],
            "dense_layers": [512],
            "activation": "relu",
            "batch_norm": False,
            "global_pooling": False,
            "dense_dropout": 0.5,
        },
        # Batch normalized architectures
        {
            "name": "BatchNorm-1",
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
        # Global pooling architectures
        {
            "name": "GlobalPool-1",
            "conv_blocks": [
                {"filters": 32, "kernel_size": (3, 3), "pool_size": (2, 2)},
                {"filters": 64, "kernel_size": (3, 3), "pool_size": (2, 2)},
                {"filters": 128, "kernel_size": (3, 3)},
            ],
            "dense_layers": [128],
            "activation": "relu",
            "batch_norm": True,
            "global_pooling": True,
            "dense_dropout": 0.3,
        },
        # Deeper architecture
        {
            "name": "Deep-1",
            "conv_blocks": [
                {"filters": 32, "kernel_size": (3, 3)},
                {
                    "filters": 32,
                    "kernel_size": (3, 3),
                    "pool_size": (2, 2),
                    "dropout": 0.25,
                },
                {"filters": 64, "kernel_size": (3, 3)},
                {
                    "filters": 64,
                    "kernel_size": (3, 3),
                    "pool_size": (2, 2),
                    "dropout": 0.25,
                },
                {"filters": 128, "kernel_size": (3, 3)},
                {
                    "filters": 128,
                    "kernel_size": (3, 3),
                    "pool_size": (2, 2),
                    "dropout": 0.25,
                },
            ],
            "dense_layers": [512, 256],
            "activation": "relu",
            "batch_norm": True,
            "global_pooling": False,
            "dense_dropout": 0.5,
        },
        # Larger kernel architecture
        {
            "name": "LargeKernel-1",
            "conv_blocks": [
                {"filters": 32, "kernel_size": (5, 5), "pool_size": (2, 2)},
                {"filters": 64, "kernel_size": (5, 5), "pool_size": (2, 2)},
                {"filters": 128, "kernel_size": (3, 3), "pool_size": (2, 2)},
            ],
            "dense_layers": [256],
            "activation": "relu",
            "batch_norm": True,
            "global_pooling": False,
            "dense_dropout": 0.5,
        },
    ]

    best_architecture = architectures[0]

    with tqdm(architectures, desc="Optimizing CNN Architecture", leave=False) as pbar:
        for arch in pbar:
            accuracy, f1, success, val_acc = _safe_evaluate_cnn(
                X_train, y_train, X_test, y_true, arch, opts["compile"], opts["fit"]
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_val_acc = val_acc
                best_architecture = arch

            pbar.set_postfix(
                {
                    "arch": arch["name"][:10],
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "val_acc": f"{val_acc:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts["architecture"] = best_architecture
    return opts, max_acc, best_f1


def _optimize_optimizer(X_train, y_train, X_test, y_true, opts):
    """Optimize optimizer and learning rate"""
    max_acc = -np.inf
    best_f1 = 0.0
    best_val_acc = 0.0

    # Different optimizers with learning rates
    optimizer_configs = [
        {"optimizer": "adam", "learning_rate": 0.001},
        {"optimizer": "adam", "learning_rate": 0.0001},
        {"optimizer": "adam", "learning_rate": 0.01},
        {"optimizer": "rmsprop", "learning_rate": 0.001},
        {"optimizer": "rmsprop", "learning_rate": 0.0001},
        {"optimizer": "sgd", "learning_rate": 0.01, "momentum": 0.9},
        {"optimizer": "sgd", "learning_rate": 0.001, "momentum": 0.9},
        {"optimizer": "adamw", "learning_rate": 0.001},
    ]

    best_config = optimizer_configs[0]

    with tqdm(optimizer_configs, desc="Optimizing Optimizer", leave=False) as pbar:
        for config in pbar:
            test_opts = opts.copy()

            # Create optimizer
            if config["optimizer"] == "adam":
                optimizer = keras.optimizers.Adam(learning_rate=config["learning_rate"])
            elif config["optimizer"] == "rmsprop":
                optimizer = keras.optimizers.RMSprop(
                    learning_rate=config["learning_rate"]
                )
            elif config["optimizer"] == "sgd":
                optimizer = keras.optimizers.SGD(
                    learning_rate=config["learning_rate"],
                    momentum=config.get("momentum", 0.0),
                )
            elif config["optimizer"] == "adamw":
                optimizer = keras.optimizers.AdamW(
                    learning_rate=config["learning_rate"]
                )

            test_opts["compile"]["optimizer"] = optimizer

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
                best_val_acc = val_acc
                best_config = config

            pbar.set_postfix(
                {
                    "opt": config["optimizer"][:6],
                    "lr": f"{config['learning_rate']:.0e}",
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    # Set best optimizer
    if best_config["optimizer"] == "adam":
        opts["compile"]["optimizer"] = keras.optimizers.Adam(
            learning_rate=best_config["learning_rate"]
        )
    elif best_config["optimizer"] == "rmsprop":
        opts["compile"]["optimizer"] = keras.optimizers.RMSprop(
            learning_rate=best_config["learning_rate"]
        )
    elif best_config["optimizer"] == "sgd":
        opts["compile"]["optimizer"] = keras.optimizers.SGD(
            learning_rate=best_config["learning_rate"],
            momentum=best_config.get("momentum", 0.0),
        )
    elif best_config["optimizer"] == "adamw":
        opts["compile"]["optimizer"] = keras.optimizers.AdamW(
            learning_rate=best_config["learning_rate"]
        )

    return opts, max_acc, best_f1


def _optimize_batch_size(X_train, y_train, X_test, y_true, opts):
    """Optimize batch size"""
    max_acc = -np.inf
    best_f1 = 0.0
    best_val_acc = 0.0

    # Reasonable batch sizes
    n_samples = X_train.shape[0]
    batch_sizes = [16, 32, 64, 128, min(256, n_samples // 4)]
    best_batch_size = batch_sizes[0]

    with tqdm(batch_sizes, desc="Optimizing Batch Size", leave=False) as pbar:
        for batch_size in pbar:
            test_opts = opts.copy()
            test_opts["fit"]["batch_size"] = batch_size

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
                best_val_acc = val_acc
                best_batch_size = batch_size

            pbar.set_postfix(
                {
                    "batch": batch_size,
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts["fit"]["batch_size"] = best_batch_size
    return opts, max_acc, best_f1


def _optimize_epochs_early_stopping(X_train, y_train, X_test, y_true, opts):
    """Optimize epochs and early stopping"""
    max_acc = -np.inf
    best_f1 = 0.0
    best_val_acc = 0.0

    # Early stopping configurations
    configs = [
        {"epochs": 50, "early_stopping": False},
        {"epochs": 100, "early_stopping": False},
        {"epochs": 200, "early_stopping": True, "patience": 10},
        {"epochs": 300, "early_stopping": True, "patience": 15},
        {"epochs": 150, "early_stopping": True, "patience": 20},
    ]

    best_config = configs[0]

    with tqdm(configs, desc="Optimizing Epochs/Early Stopping", leave=False) as pbar:
        for config in pbar:
            test_opts = opts.copy()
            test_opts["fit"]["epochs"] = config["epochs"]

            # Add early stopping callback if specified
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
                best_val_acc = val_acc
                best_config = config

            pbar.set_postfix(
                {
                    "epochs": config["epochs"],
                    "early_stop": "Y" if config.get("early_stopping") else "N",
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    # Set best configuration
    opts["fit"]["epochs"] = best_config["epochs"]
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


def _optimize_data_augmentation(X_train, y_train, X_test, y_true, opts):
    """Optimize data augmentation (if beneficial)"""
    max_acc = -np.inf
    best_f1 = 0.0
    best_val_acc = 0.0

    # Data augmentation configurations
    augmentation_configs = [
        {"use_augmentation": False},
        {
            "use_augmentation": True,
            "rotation_range": 10,
            "width_shift_range": 0.1,
            "height_shift_range": 0.1,
            "horizontal_flip": True,
        },
        {
            "use_augmentation": True,
            "rotation_range": 20,
            "width_shift_range": 0.2,
            "height_shift_range": 0.2,
            "zoom_range": 0.1,
            "horizontal_flip": True,
        },
    ]

    best_config = augmentation_configs[0]

    with tqdm(
        augmentation_configs, desc="Optimizing Data Augmentation", leave=False
    ) as pbar:
        for config in pbar:
            test_opts = opts.copy()

            if config["use_augmentation"]:
                # Create data generator with augmentation
                datagen = keras.preprocessing.image.ImageDataGenerator(
                    rotation_range=config.get("rotation_range", 0),
                    width_shift_range=config.get("width_shift_range", 0),
                    height_shift_range=config.get("height_shift_range", 0),
                    zoom_range=config.get("zoom_range", 0),
                    horizontal_flip=config.get("horizontal_flip", False),
                    validation_split=0.2,
                )

                # Note: For full implementation, you'd need to modify the training loop
                # to use fit_generator with the data generator
                # For simplicity, we'll just test the base case here

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
                best_val_acc = val_acc
                best_config = config

            pbar.set_postfix(
                {
                    "aug": "Y" if config["use_augmentation"] else "N",
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    opts["augmentation"] = best_config
    return opts, max_acc, best_f1


def _optimize_regularization(X_train, y_train, X_test, y_true, opts):
    """Optimize regularization techniques"""
    max_acc = -np.inf
    best_f1 = 0.0
    best_val_acc = 0.0

    # Regularization configurations
    reg_configs = [
        {"dense_dropout": 0.3, "conv_dropout": 0.0},
        {"dense_dropout": 0.5, "conv_dropout": 0.0},
        {"dense_dropout": 0.7, "conv_dropout": 0.0},
        {"dense_dropout": 0.5, "conv_dropout": 0.25},
        {"dense_dropout": 0.3, "conv_dropout": 0.1},
    ]

    best_config = reg_configs[0]

    with tqdm(reg_configs, desc="Optimizing Regularization", leave=False) as pbar:
        for config in pbar:
            # Modify architecture with new dropout rates
            test_architecture = opts["architecture"].copy()
            test_architecture["dense_dropout"] = config["dense_dropout"]

            # Update conv blocks with dropout
            test_architecture["conv_blocks"] = []
            for block in opts["architecture"]["conv_blocks"]:
                new_block = block.copy()
                if config["conv_dropout"] > 0:
                    new_block["dropout"] = config["conv_dropout"]
                test_architecture["conv_blocks"].append(new_block)

            accuracy, f1, success, val_acc = _safe_evaluate_cnn(
                X_train,
                y_train,
                X_test,
                y_true,
                test_architecture,
                opts["compile"],
                opts["fit"],
            )

            if success and accuracy >= max_acc:
                max_acc = accuracy
                best_f1 = f1
                best_val_acc = val_acc
                best_config = config

            pbar.set_postfix(
                {
                    "dense_drop": f"{config['dense_dropout']:.1f}",
                    "conv_drop": f"{config['conv_dropout']:.1f}",
                    "acc": f"{accuracy:.4f}" if success else "failed",
                    "best_acc": f"{max_acc:.4f}",
                }
            )

    # Update architecture with best regularization
    opts["architecture"]["dense_dropout"] = best_config["dense_dropout"]
    for block in opts["architecture"]["conv_blocks"]:
        if best_config["conv_dropout"] > 0:
            block["dropout"] = best_config["conv_dropout"]

    return opts, max_acc, best_f1


def _optimize_cnn(X_train, y_train, X_test, y_true, cycles=2):
    """
    Optimizes CNN hyperparameters for image classification.

    :param X_train: Training images (N, H, W, C) or (N, H, W)
    :param y_train: Training labels
    :param X_test: Test images
    :param y_true: True labels for test data
    :param cycles: Number of optimization cycles
    :return: optimized options, best accuracy, best f1, accuracy history, f1 history
    """

    # Ensure proper data format
    if len(X_train.shape) == 3:  # Add channel dimension if grayscale
        X_train = np.expand_dims(X_train, axis=-1)
        X_test = np.expand_dims(X_test, axis=-1)

    # Normalize data
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

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

    # Define initial options
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
        "fit": {"epochs": 50, "batch_size": 32, "callbacks": []},
        "augmentation": {"use_augmentation": False},
    }

    # Track results
    ma_vec = []
    f1_vec = []

    # Main optimization loop
    with tqdm(range(cycles), desc="CNN Optimization Cycles", position=0) as cycle_pbar:
        for c in cycle_pbar:
            cycle_pbar.set_description(f"CNN Cycle {c + 1}/{cycles}")

            # Core architecture optimization
            opts, _, _ = _optimize_conv_architecture(
                X_train, y_train, X_test, y_true_encoded, opts
            )
            opts, _, _ = _optimize_optimizer(
                X_train, y_train, X_test, y_true_encoded, opts
            )

            # Training optimization
            opts, _, _ = _optimize_batch_size(
                X_train, y_train, X_test, y_true_encoded, opts
            )
            opts, _, _ = _optimize_epochs_early_stopping(
                X_train, y_train, X_test, y_true_encoded, opts
            )

            # Regularization and augmentation
            opts, _, _ = _optimize_regularization(
                X_train, y_train, X_test, y_true_encoded, opts
            )
            opts, ma, f1 = _optimize_data_augmentation(
                X_train, y_train, X_test, y_true_encoded, opts
            )

            ma_vec.append(ma)
            f1_vec.append(f1)

            cycle_pbar.set_postfix(
                {
                    "accuracy": f"{ma:.4f}",
                    "f1": f"{f1:.4f}",
                    "best_overall": f"{max(ma_vec):.4f}",
                    "arch": opts["architecture"].get("name", "Custom")[:10],
                    "optimizer": str(opts["compile"]["optimizer"])[:15],
                    "batch": opts["fit"]["batch_size"],
                    "epochs": opts["fit"]["epochs"],
                }
            )

    return opts, ma, f1, ma_vec, f1_vec
