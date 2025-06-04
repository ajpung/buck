# Import necessary libraries
import gc
import os
from glob import glob
from typing import List, Tuple, Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from buck.data.image_processing import make_bw_square, resize_image

# Find/ingest files in folder; force square & b/w
files = glob("..\\..\\..\\images\\squared\\*.png")
files = [s for s in files if "xpx" not in s]

import numpy as np
import cv2


def extract_fast_numpy_features(X_images):
    n_samples = X_images.shape[0]

    # Check if color or grayscale
    if len(X_images.shape) == 4 and X_images.shape[3] == 3:
        # Color images: 64×64×3 + 32×32×3 = 15360 features
        feature_size = 12288 + 3072
    else:
        # Grayscale images: 64×64×1 + 32×32×1 = 5120 features
        feature_size = 4096 + 1024

    all_features = np.zeros((n_samples, feature_size))

    for i in range(n_samples):
        img = X_images[i]

        # Resize to two different scales
        img_64 = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
        img_32 = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)

        # Flatten
        features_64 = img_64.flatten()
        features_32 = img_32.flatten()

        all_features[i] = np.concatenate([features_64, features_32])

    return all_features


def extract_multi_resolution_features_grayscale(X_images):
    """
    Extract multi-resolution features from square grayscale images
    No cropping needed - uses full image at multiple scales

    Args:
        X_images: Grayscale image array of shape (N, H, W, 1)

    Returns:
        features: Multi-resolution feature array (N, total_features)
    """

    print(f"Extracting multi-resolution features from {X_images.shape}...")

    all_features = []

    for i, img in enumerate(X_images):
        # Squeeze to 2D for cv2.resize
        img_2d = np.squeeze(img, axis=-1)  # (288, 288, 1) -> (288, 288)

        # Convert to uint8 if needed
        if img_2d.dtype != np.uint8:
            img_uint8 = (
                (img_2d * 255).astype(np.uint8)
                if img_2d.max() <= 1.0
                else img_2d.astype(np.uint8)
            )
        else:
            img_uint8 = img_2d

        # RESOLUTION 1: High detail (96x96) - captures fine morphology
        high_res = cv2.resize(img_uint8, (96, 96))
        high_features = high_res.flatten()  # 96*96 = 9,216 features

        # RESOLUTION 2: Medium detail (48x48) - overall body proportions
        medium_res = cv2.resize(img_uint8, (48, 48))
        medium_features = medium_res.flatten()  # 48*48 = 2,304 features

        # RESOLUTION 3: Low detail (24x24) - general body shape
        low_res = cv2.resize(img_uint8, (24, 24))
        low_features = low_res.flatten()  # 24*24 = 576 features

        # RESOLUTION 4: Very low detail (12x12) - basic proportions
        context_res = cv2.resize(img_uint8, (12, 12))
        context_features = context_res.flatten()  # 12*12 = 144 features

        # Combine all resolution features
        combined_features = np.concatenate(
            [
                high_features,  # Fine deer morphology details
                medium_features,  # Body proportions and structure
                low_features,  # General shape and pose
                context_features,  # Basic size/proportion ratios
            ]
        )

        all_features.append(combined_features)

    # Convert to numpy array and normalize
    feature_array = np.array(all_features, dtype=np.float32)
    feature_array = feature_array / 255.0  # Normalize to 0-1

    return feature_array


def extract_multi_resolution_features(X_images):
    """
    Extract multi-resolution features from images for better deer morphology analysis

    Args:
        X_images: Image array of shape (N, H, W, C)

    Returns:
        features: Multi-resolution feature array (N, total_features)
    """

    print(f"Extracting multi-resolution features from {X_images.shape}...")

    all_features = []

    for i, img in enumerate(X_images):
        # Convert to uint8 if needed for cv2.resize
        if img.dtype != np.uint8:
            img_uint8 = (
                (img * 255).astype(np.uint8)
                if img.max() <= 1.0
                else img.astype(np.uint8)
            )
        else:
            img_uint8 = img

        # RESOLUTION 1: High-detail center crop (deer body focus)
        # Extract center 144x144 region where deer body is usually located
        h, w = img_uint8.shape[:2]
        center_start_h = (h - 144) // 2
        center_start_w = (w - 144) // 2
        center_crop = img_uint8[
            center_start_h : center_start_h + 144,
            center_start_w : center_start_w + 144,
            :,
        ]

        # Subsample center crop to 72x72 (good detail, manageable size)
        center_resized = cv2.resize(center_crop, (72, 72))
        center_features = center_resized.flatten()  # 72*72*3 = 15,552 features

        # RESOLUTION 2: Medium resolution full image (overall proportions)
        # Resize full image to 48x48 to capture overall body proportions
        medium_resized = cv2.resize(img_uint8, (48, 48))
        medium_features = medium_resized.flatten()  # 48*48*3 = 6,912 features

        # RESOLUTION 3: Low resolution full image (general shape)
        # Resize to 24x24 for overall deer shape and pose
        low_resized = cv2.resize(img_uint8, (24, 24))
        low_features = low_resized.flatten()  # 24*24*3 = 1,728 features

        # RESOLUTION 4: Very low resolution for global context
        # Resize to 12x12 for very basic shape information
        context_resized = cv2.resize(img_uint8, (12, 12))
        context_features = context_resized.flatten()  # 12*12*3 = 432 features

        # Combine all resolution features
        combined_features = np.concatenate(
            [
                center_features,  # Fine details of deer body
                medium_features,  # Overall proportions
                low_features,  # General shape
                context_features,  # Global context
            ]
        )

        all_features.append(combined_features)

    # Convert to numpy array
    feature_array = np.array(all_features, dtype=np.float32)

    # Normalize to 0-1 range
    feature_array = feature_array / 255.0

    return feature_array


def extract_fast_multi_resolution_features(X_images):
    """
    Faster version with fewer features but still multi-scale
    """

    print(f"Extracting FAST multi-resolution features from {X_images.shape}...")

    all_features = []

    for i, img in enumerate(X_images):

        # Convert to uint8 if needed
        if img.dtype != np.uint8:
            img_uint8 = (
                (img * 255).astype(np.uint8)
                if img.max() <= 1.0
                else img.astype(np.uint8)
            )
        else:
            img_uint8 = img

        # RESOLUTION 1: Center crop at medium resolution
        h, w = img_uint8.shape[:2]
        center_start_h = (h - 112) // 2
        center_start_w = (w - 112) // 2
        center_crop = img_uint8[
            center_start_h : center_start_h + 112,
            center_start_w : center_start_w + 112,
            :,
        ]
        center_resized = cv2.resize(center_crop, (56, 56))  # 56*56*3 = 9,408
        center_features = center_resized.flatten()

        # RESOLUTION 2: Full image at low resolution
        full_resized = cv2.resize(img_uint8, (28, 28))  # 28*28*3 = 2,352
        full_features = full_resized.flatten()

        # Combine features
        combined_features = np.concatenate([center_features, full_features])
        all_features.append(combined_features)

    feature_array = np.array(all_features, dtype=np.float32) / 255.0

    total_features = 9408 + 2352  # = 11,760 features

    print(f"✅ Fast multi-resolution extraction complete!")
    print(f"   Total features: {total_features:,}")
    print(f"   Final shape: {feature_array.shape}")

    return feature_array


def prepare_optimal_deer_features(X_train, X_test):
    """Optimal balance of accuracy and speed for deer morphology"""
    print("Applying center crop + light subsample...")
    # Focus on center 144x144 where deer body usually is
    center_train = X_train[:, 40:184, 40:184, :]
    center_test = X_test[:, 40:184, 40:184, :]
    # Light subsample (every 2nd pixel) to 72x72
    sub_train = center_train[:, ::2, ::2, :]
    sub_test = center_test[:, ::2, ::2, :]
    # Flatten
    X_train_flat = sub_train.reshape(sub_train.shape[0], -1)
    X_test_flat = sub_test.reshape(sub_test.shape[0], -1)
    return X_train_flat, X_test_flat


# Reshape / concatenate images
def ingest_resize_stack(files: list) -> list[np.ndarray[Any, Any]]:
    """
    Ingests images, reshapes them to square, and concatenates them into a 3D array.
    :param files: List of file paths
    :return images: 3D numpy array of images
    """
    images = []
    for img in files:
        img_sqrbw = make_bw_square(img)
        img_resize = resize_image(img_sqrbw, new_dim=288)  # type: ignore
        images.append(img_resize)
    images = np.array(images)  # type: ignore
    return images


def extract_labels(
    files: list,
) -> tuple[list[Any], list[Any], list[Any], list[float], list[Any]]:
    """
    Extracts labels from file names.
    :param files: List of file paths
    :return labels: List of labels
    """
    found = []  # Date the data was acquired
    taken = []  # Date the image was taken
    state = []  # State the image was taken in
    ages = []  # Age of the deer
    provider = []  # Image provider
    for file in files:
        cfound = file.split("\\")[-1].split(".")[0].split("_")[0]
        found.append(cfound)
        ctaken = file.split("\\")[-1].split(".")[0].split("_")[1]
        taken.append(ctaken)
        cstate = file.split("\\")[-1].split(".")[0].split("_")[2]
        state.append(cstate)
        curage = float(
            file.split("\\")[-1].split(".")[0].split("_")[3].replace("p", ".")
        )
        ages.append(curage)
        cprovider = file.split("\\")[-1].split(".")[0].split("_")[4]
        provider.append(cprovider)
    return found, taken, state, ages, provider


def ingest_images(fpath: str):  # type: ignore
    """
    Ingests images from a given file path, resizes them to square, and extracts labels.

    :param fpath: File path to the images
    :return: Tuple containing the images as a 3D numpy array and a list of ages
    """

    # Avoid Tensorflow warnings
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    # Find/ingest files in folder; force square & b/w
    files = glob(fpath)
    files = [s for s in files if "xpx" not in s]

    # Ingest images
    images = ingest_resize_stack(files)
    _, _, _, ages, _ = extract_labels(files)
    return images, ages


def augment_class(class_images, class_labels, target_count, batch_size=16):
    """Augment a class of images in small batches to avoid memory issues."""
    samples_to_generate = target_count - len(class_images)
    if samples_to_generate <= 0:
        return class_images, class_labels  # No augmentation needed

    # Initialize the data generator
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    # Create lists to collect augmented data
    all_aug_images = []
    all_aug_labels = []
    remaining = samples_to_generate

    # Process in small batches
    orig_batch_size = min(batch_size, len(class_images))

    while remaining > 0:
        # Take a small batch of original images
        indices = np.random.choice(len(class_images), orig_batch_size, replace=True)
        batch_imgs = class_images[indices]
        batch_lbls = class_labels[indices]

        # Generate one batch of augmented images
        aug_gen = datagen.flow(
            batch_imgs, batch_lbls, batch_size=orig_batch_size, shuffle=True
        )
        aug_imgs, aug_lbls = next(aug_gen)

        # Take only what we need
        batch_to_take = min(orig_batch_size, remaining)
        all_aug_images.append(aug_imgs[:batch_to_take])
        all_aug_labels.append(aug_lbls[:batch_to_take])

        # Update remaining count
        remaining -= batch_to_take

    # Combine all batches
    aug_images = np.concatenate(all_aug_images)
    aug_labels = np.concatenate(all_aug_labels)

    # Combine with original data
    combined_images = np.concatenate([class_images, aug_images])
    combined_labels = np.concatenate([class_labels, aug_labels])

    return combined_images, combined_labels


def split_data(images: List, ages: List) -> Tuple:
    """
    Splits the data into training, validation, and test sets.

    :param images: Array of images being processed
    :param ages: List of ages for each image
    :return: Tuple of training, validation, and test sets
    """
    # Map 5.5+ years into a single "mature" class
    ages_array = np.array(ages)
    mature_ages = []

    for i, age in enumerate(ages_array):
        if age >= 5.5:
            ages_array[i] = 5.5  # Set all ages 5.5+ to 5.5
            if age not in mature_ages:
                mature_ages.append(age)

    # Now create your label mapping with the modified ages
    label_mapping = {label: i for i, label in enumerate(np.unique(ages_array))}

    # Apply the mapping to convert labels to integers
    integer_labels = np.array([label_mapping[l] for l in ages_array])

    # Implement a stratified split to ensure all classes appear in train and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        images,
        integer_labels,
        test_size=0.2,
        random_state=42,
        stratify=integer_labels,  # This ensures proportional representation of classes
    )

    # One-hot encode labels AFTER splitting but BEFORE the next split
    num_classes = len(label_mapping)
    y_train_val_onehot = keras.utils.to_categorical(y_train_val, num_classes)
    y_test_onehot = keras.utils.to_categorical(y_test, num_classes)

    # Normalize and reshape images
    X_train_val = X_train_val.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0
    X_train_val = X_train_val.reshape(X_train_val.shape[0], 288, 288, 3)
    X_test = X_test.reshape(X_test.shape[0], 288, 288, 3)

    # Implement stratified split for validation too
    X_train_orig, X_valid, y_train_orig, y_valid = train_test_split(
        X_train_val,
        y_train_val_onehot,
        test_size=0.2,
        random_state=42,
        stratify=np.argmax(y_train_val_onehot, axis=1),  # Stratify by class
    )
    print(X_train_orig.shape, X_valid.shape, X_test.shape)

    return (
        X_train_orig,
        y_train_orig,
        X_valid,
        y_valid,
        X_test,
        y_test_onehot,
        ages_array,
        label_mapping,
    )


def homogenize_data(
    X_train_orig, y_train_orig, X_test, y_test_onehot, label_mapping, aug_mult=2
) -> Tuple:
    # Ensure we have the correct mappings
    reverse_mapping = {i: label for label, i in label_mapping.items()}
    num_classes = len(reverse_mapping)

    # Calculate the maximum count among all classes in the original training data
    unique_classes = np.unique(np.argmax(y_train_orig, axis=1))
    class_counts = []
    for class_idx in unique_classes:
        class_count = np.sum(np.argmax(y_train_orig, axis=1) == class_idx)
        class_counts.append(class_count)
    max_count = max(class_counts)

    # Calculate target count per class
    target_per_class = max_count * aug_mult

    # Create empty lists for augmented data
    X_train_aug_list = []
    y_train_aug_list = []

    # Process each class separately
    for class_idx in unique_classes:
        # Get indices of samples from this class
        class_indices = np.where(np.argmax(y_train_orig, axis=1) == class_idx)[0]
        class_count = len(class_indices)

        # Get the class samples
        class_images = X_train_orig[class_indices]
        class_labels = y_train_orig[class_indices]

        # Augment this class efficiently
        print(
            f"  Class {class_idx} (Age {reverse_mapping[class_idx]}): {class_count} → {target_per_class} samples"
        )
        aug_images, aug_labels = augment_class(
            class_images, class_labels, target_per_class, batch_size=16
        )

        # Add to our collections
        X_train_aug_list.append(aug_images)
        y_train_aug_list.append(aug_labels)

        # Force garbage collection after each class
        gc.collect()

    # Combine all classes
    X_train_augmented = np.concatenate(X_train_aug_list)
    y_train_augmented = np.concatenate(y_train_aug_list)

    # Convert labels to integers
    y_train_flat = np.argmax(y_train_augmented, axis=1)
    y_true = np.argmax(y_test_onehot, axis=1)

    """
    # For EfficientNet
    X_train_flat = extract_fast_numpy_features(X_train_augmented)
    X_test_flat = extract_fast_numpy_features(X_test)
    # Apply standardization
    scaler = StandardScaler()
    X_train_final = scaler.fit_transform(X_train_flat)
    X_test_final = scaler.transform(X_test_flat)
    # Free up memory (delete the unscaled versions)
    del X_train_flat, X_test_flat
    gc.collect()
    """

    # Return the final scaled versions
    return X_train_augmented, y_train_flat, X_test, y_true, label_mapping, num_classes
