from typing import Any

import cv2
import numpy as np
from matplotlib.image import imread


def make_bw_square(file: str) -> np.ndarray[tuple[int, ...], Any]:
    """
    Ingests an image file, reads dimensions, and forces image to be a black and white square image.
    :param file: Path to the image file
    :return new_image: Image as a numpy array
    """
    # Ingest images
    image = imread(file)
    px, py, pz = image.shape
    # Find minimum value between X and Y dimensions
    min_dim = min(px, py)
    # If data are 3D, flatten to black and white
    if pz > 2:
        new_image = image[:, :, 0]
    else:
        new_image = image

    # Force image to be square
    new_image = new_image[:min_dim, :min_dim]
    return new_image


def resize_image(image: np.ndarray, new_dim: int) -> np.ndarray:
    """
    Resizes the image to a square with the given minimum dimension.
    :param image: Image as a numpy array
    :param min_dim: Minimum dimension for resizing
    :return new_image: Resized image as a numpy array
    """
    # Define dimensions of new image
    new_dimensions = (new_dim, new_dim)
    # Resize the image using interpolation
    resized_img = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)
    return resized_img
