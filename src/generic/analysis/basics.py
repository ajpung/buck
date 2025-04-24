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
