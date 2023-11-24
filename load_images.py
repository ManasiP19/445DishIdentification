import os
import pandas as pd

# funtion unzips the images.zip file and creates a list of tuples that hold the image and its label (dish name)
def load_imagesp(directory_path):
    with open('classes.txt', 'r') as classes_file:
        folder_name = [line.strip() for line in classes_file]
    
    with open('labels.txt', 'r') as labels_file:
        dish_name = [line.strip() for line in labels_file]
    
    mapping = dict(zip(folder_name, dish_name))

    image_paths = []

    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.lower().edswith('jpg'):
                folder_name = os.path.basename(root)
                dish_name = mapping.get(folder_name, 'Unkown')
                full_image_path = os.path.join(root, file)
                image_paths.append((full_image_path, dish_name))
    df = pd.DataFrame(image_paths, columns=['image_path', 'labels'])
    return df
