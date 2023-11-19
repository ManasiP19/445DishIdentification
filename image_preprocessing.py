# -*- coding: utf-8 -*-
import cv2
import matplotlib.pylab as plt
import numpy as np

# define preprocessing function 
def image_preprocessing(image, normalize_pixel_vals = None, flattening = None, new_dimensions = None, sharpening = None, resize_factor = None, to_greyscale = None):
    
    # initialize image for preprocessing
    preprocessed_image = cv2.imread(image)

    # normalize pixel values
    if normalize_pixel_vals == True:
        preprocessed_image / 255
    
    # perform flattening
    if flattening == True:
        preprocessed_image.flatten()
    
    # force image to new dimensions
    if new_dimensions != None:
        preprocessed_image = cv2.resize(preprocessed_image, (new_dimensions[0], new_dimensions[1]))
    
    # perform image sharpening
    if sharpening == True:
        kernel_sharpening = np.array([[-1, -1, -1],
                                      [-1, 9, -1],
                                      [-1, -1, -1]])
        
        preprocessed_image = cv2.filter2D(preprocessed_image, -1, kernel_sharpening)
    
    # resize image
    if resize_factor != None:
        preprocessed_image = cv2.resize(preprocessed_image, None, fx = resize_factor, fy = resize_factor)
    
    # convert image to greyscale
    if to_greyscale == True:
        preprocessed_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2GRAY)
    
    # do other things? such as ZCA whitening / PCA 
    
    return preprocessed_image