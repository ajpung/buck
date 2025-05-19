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
    X_train_val = X_train_val.reshape(X_train_val.shape[0], 288, 288, 1)
    X_test = X_test.reshape(X_test.shape[0], 288, 288, 1)

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

    # Flatten the data - using full images
    X_train_flat = X_train_augmented.reshape(X_train_augmented.shape[0], -1).astype(
        np.float32
    )
    X_test_flat = X_test.reshape(X_test.shape[0], -1).astype(np.float32)

    # Apply standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_test_scaled = scaler.transform(X_test_flat)

    # Free up memory
    del X_train_flat, X_test_flat
    gc.collect()

    # Apply PCA for dimensionality reduction
    n_components = min(20, X_train_scaled.shape[0] - 5, X_train_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # Free up memory
    del X_train_scaled, X_test_scaled
    gc.collect()

    return X_train_pca, y_train_flat, X_test_pca, y_true, label_mapping, num_classes
