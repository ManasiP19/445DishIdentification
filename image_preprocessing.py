import cv2
import matplotlib.pylab as plt
import numpy as np
from PIL import Image

# define preprocessing function 
def image_preprocessing(image, normalize_pixel_vals=None, flattening=None, new_dimensions=None,
                         sharpening=None, resize_factor=None, to_greyscale=None):
    # Check if the input is a file path or a PIL Image object
    if isinstance(image, str):
        # File path provided, read the image using cv2.imread
        preprocessed_image = cv2.imread(image)
    elif isinstance(image, Image.Image):
        # PIL Image object provided, convert it to an array
        preprocessed_image = np.array(image)
    else:
        raise ValueError("Unsupported input type. Provide either a file path or a PIL Image object.")

    print(f"Image shape before preprocessing: {preprocessed_image.shape}")

    # normalize pixel values
    if normalize_pixel_vals:
        preprocessed_image = preprocessed_image.astype('float32')  # Convert to float32
        preprocessed_image /= 255.0

    # perform flattening
    if flattening:
        preprocessed_image = preprocessed_image.flatten()

    # force image to new dimensions
    if new_dimensions:
        preprocessed_image = cv2.resize(preprocessed_image, (new_dimensions[0], new_dimensions[1]))

    print(f"Image shape after resizing: {preprocessed_image.shape}")

    # perform image sharpening
    if sharpening:
        kernel_sharpening = np.array([[-1, -1, -1],
                                      [-1, 9, -1],
                                      [-1, -1, -1]])
        preprocessed_image = cv2.filter2D(preprocessed_image, -1, kernel_sharpening)

    # resize image
    if resize_factor:
        preprocessed_image = cv2.resize(preprocessed_image, None, fx=resize_factor, fy=resize_factor)

    # convert image to greyscale if it has more than one channel
    if to_greyscale and preprocessed_image.ndim == 3 and preprocessed_image.shape[-1] > 1:
        preprocessed_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2GRAY)

    print(f"Image shape after conversion to greyscale: {preprocessed_image.shape}")

    # do other things? such as ZCA whitening / PCA

    return preprocessed_image
