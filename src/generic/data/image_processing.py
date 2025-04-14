from glob import glob

import cv2
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread


def make_bw_square(file: str) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """
    Ingests an image file, reads dimensions, and forces image to be a black and white square image.
    :param file: Path to the image file
    :return new_image: Image as a numpy array
    """
    # Ingest images
    image = imread(cur_img)
    px, py, pz = image.shape
    # Find minimum value between X and Y dimensions
    min_dim = min(px, py)
    print(px, py, pz)
    # If data are 3D, flatten to black and white
    if pz > 2:
        new_image = image[:, :, 0]
    else:
        new_image = image

    # Force image to be square
    new_image = new_image[:min_dim, :min_dim]
    return image, new_image


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


# Find/ingest files in folder; force square & b/w
files = glob("..\..\..\images\squared\*.png")
for cur_img in files:
    image_0, image_1 = make_bw_square(cur_img)
    image_2 = resize_image(image_1, new_dim=288)

# Demonstrate processing
plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(image_0)
plt.subplot(1, 3, 2)
plt.imshow(image_1, cmap="gray")
plt.subplot(1, 3, 3)
plt.imshow(image_2, cmap="gray")
plt.show()
