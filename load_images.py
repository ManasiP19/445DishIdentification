from PIL import Image
import zipfile
import os
import pandas as pd
from tensorflow.keras.preprocessing.image import img_to_array

# funtion unzips the images.zip file and creates a list of tuples that hold the image and its label (dish name)
def load_images_from_zip(zip_path):
    with open('classes.txt', 'r') as classes_file:
        folder_names = [line.strip() for line in classes_file]

    with open('labels.txt', 'r') as labels_file:
        dish_names = [line.strip() for line in labels_file]

    mapping = dict(zip(folder_names, dish_names))

    image_paths = []

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        image_files = [name for name in zip_ref.namelist() if name.endswith('jpg')]

        for image_file in image_files:
            folder_name = os.path.dirname(image_file)
            dish_name = mapping.get(folder_name, 'Unknown')
            # Construct the correct full path within the zip file
            full_image_path = image_file
            image_paths.append((full_image_path, dish_name))

    df = pd.DataFrame(image_paths, columns=['image_path', 'labels'])
    return df
