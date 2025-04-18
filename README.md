<p align="center">
    <img src="docs/_static/logo.png" alt="BUCK Logo" width="300"/>
</p>
<h1 align="center">
    BUCK
</h1>
<h2 align="center">
    Biological Understanding via Computer Knowledge
</h2>


# Introduction
A tool for predicting the age of male deer based on National Deer Association
(NDA)-provided images and professional ratings. Specifically, this analysis follows the textbook. "Deep Learning for Vision Systems", by M. Elgendy.

# Buck aging
## Introduction
There are many ways to predict age on a male deer. The most common method is to
look at the teeth, but this requires a professional to examine the animal. Other
techniques have been deeply investigated by Lindsay Thomas Jr. of the National
Deer Association (NDA) including antler characteristics (mass, size, etc.) as
well as body proportions.

## Antler characteristics
Lindsay's studies have shown an increase in average antler mass as a function of
age. This is a good indicator of age, but it requires a professional to examine the
sheds. Furthermore, the curves are nonlinear and vary by region. This means that
the model must be trained on a specific region to be accurate, and this is not
practical for the average hunter. Other characteristics include average beam length,
circumference, and tine length, but these suffer from the same issues as mass.

## Supervised Learning
### BUCK
Alternatively, the NDA has provided a set of images and ratings for a number of
deer, which can be used to train a model to predict age based on images alone.
This is the method we will use in this project BUCK (Biological Understanding
via Computer Knowledge). Images of the deer were taken from the NDA website and
the ratings were taken from the NDA's "Age This!" articles.

To perform its analysis, BUCK uses a convolutional neural network (CNN) to extract
features from the images. The CNN is a type of deep learning model that is
particularly well-suited for image classification tasks. The model is trained on
a dataset of images and their corresponding age ratings, and learns to
recognize patterns in the images that are indicative of age. Once trained, the
model can be used to predict the age of new images of deer.

Images gathered from the website were sized and cropped, making sure to include the
full deer's body; for non-square images, padding was added to the top or bottom of
the image to make it square. The images were then resized to 224x224 pixels and
normalized to a range of 0-1. The images were then split into training and test
sets, with 80% of the images used for training and 20% for testing. The training
set was then augmented using random rotations, flips, and brightness adjustments
to increase the size of the training set and improve the model's performance.

# Installation
```
# Create a virtual environment
python -m venv buck-env
d# Activate the virtual environment
.\buck-env\Scripts\activate
# Install the package in editable mode
python -m pip install -e .
# Install the environment in Jupyter
python -m ipykernel install --user --name=buck-env --display-name="BUCK Environment"
```