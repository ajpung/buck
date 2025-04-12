import os
import numpy as np
from glob import glob
from matplotlib.image import imread

# Find images in folder
# def find_image_shapes(folder:str) -> ist:
min_val = np.inf

# Find/ingest files in folder, find minimum global dimension
files = glob("..\..\..\images\squared\*.png")
for cur_img in files:
    # Ingest images
    image = imread(cur_img)
    px, py, pz = image.shape
    # Find minimum value between X and Y dimensions
    min_dim = min(px, py)
    if min_dim < min_val:
        min_val = min_dim
