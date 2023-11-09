from PIL import Image
import zipfile
import os
import pandas as pd

# funtion unzips the images.zip file and creates a list of tuples that hold the image and its label (dish name)
def load_images_from_zip(zip_path):
    # read list of folder names from the classes.txt file
    with open('classes.txt', 'r') as classes_file:
        folder_names = [line.strip() for line in classes_file]

    # read list of dish names from labels.txt file
    with open('labels.txt', 'r') as labels_file:
        dish_names = [line.strip() for line in labels_file]

    # create mapping between folder names and dish names for labelling
    mapping = dict(zip(folder_names, dish_names))

    images_labels = []

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        image_files = [name for name in zip_ref.namelist() if name.endswith('jpg')]

        for image_file in image_files:
            with zip_ref.open(image_file) as file:
                img = Image.open(file)

                folder_name = os.path.dirname(image_file)

                dish_name = mapping.get(folder_name, 'Unknown')

                images_labels.append((img, dish_name))
            
    df = pd.DataFrame(images_labels)
    df.columns = ['images', 'labels']

    return df