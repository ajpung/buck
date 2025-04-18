import keras
import numpy as np
from glob import glob
from typing import Any
from sklearn.model_selection import train_test_split
from generic.data.image_processing import make_bw_square, resize_image

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


def extract_labels(files: list) -> list:
    """
    Extracts labels from file names.
    :param files: List of file paths
    :return labels: List of labels
    """
    labels = []
    for file in files:
        label = float(
            file.split("\\")[-1].split(".")[0].split("_")[-1].replace("p", ".")
        )
        labels.append(label)
    return labels
