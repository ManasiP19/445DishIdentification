import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from skimage.feature import hog
from skimage import exposure
from load_images import load_images_from_zip
import zipfile
from image_preprocessing import image_preprocessing

def preprocess_and_extract_hog(img):
    img = image_preprocessing(img, normalize_pixel_vals=True, to_greyscale=True)
    img = img.resize((64, 64))  # Resize for consistency
    fd, hog_image = hog(np.array(img), orientations=8, pixels_per_cell=(4, 4), cells_per_block=(1, 1), visualize=True)
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    return fd

def extract_features_and_labels(dataframe, zip_file):
    features = []
    labels = []

    for index, row in dataframe.iterrows():
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            with zip_ref.open(row['image_path']) as file:
                img = Image.open(file)
                hog_features = preprocess_and_extract_hog(img)
                features.append(hog_features)
                labels.append(row['labels_encoded'])

    return np.array(features), np.array(labels)

# Load images and labels
dataset_path = 'images.zip'
df = load_images_from_zip(dataset_path)

# Encode labels
label_encoder = LabelEncoder()
df['labels_encoded'] = label_encoder.fit_transform(df['labels'])
num_classes = len(label_encoder.classes_)

# Split the dataset into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Extract HOG features for training set
train_features, train_labels = extract_features_and_labels(train_df, 'images.zip')

# Extract HOG features for test set
test_features, test_labels = extract_features_and_labels(test_df, 'images.zip')

# Standardize the features
scaler = StandardScaler()
train_features_std = scaler.fit_transform(train_features)
test_features_std = scaler.transform(test_features)

# Train an SVM model
svm_model = SVC(kernel='linear', C=1.0)
svm_model.fit(train_features_std, train_labels)

# Make predictions on the test set
svm_predictions = svm_model.predict(test_features_std)

# Evaluate the SVM model
accuracy = accuracy_score(test_labels, svm_predictions)
print(f'SVM Accuracy: {accuracy}')
