import keras
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
from generic.data.image_processing import make_bw_square, resize_image

# Find/ingest files in folder; force square & b/w
files = glob("..\..\..\images\squared\*.png")

# Reshape / concatenate images
for img in files:
    img_sqrbw = make_bw_square(img)
    img_resize = resize_image(img_sqrbw, new_dim=288)  # type: ignore
    # Concatenate all images
    if img == files[0]:
        images = img_resize
    else:
        images = np.concatenate((images, img_resize), axis=0)

## Split test & train data via Sklearn
# X_train, X_test, y_train, y_test = train_test_split(combined_images, labels, test_size=0.2, random_state=42)

print(images.shape)
