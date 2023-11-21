import pandas as pd
import numpy as np
import tensorflow_cnn as tf
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model, saved_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from load_images import load_images_from_zip
import zipfile
from image_preprocessing import image_preprocessing
import os

df = load_images_from_zip('images.zip')
df.head()

label_encoder = LabelEncoder()
df['labels_encoded'] = label_encoder.fit_transform(df['labels'])
num_classes = len(label_encoder.classes_)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)


def custom_data_generator(dataframe, batch_size, img_size, preprocessing_params=None):
    while True:
        for i in range(0, len(dataframe), batch_size):
            batch_df = dataframe.iloc[i:i + batch_size]
            batch_images = []
            batch_labels = []
            for index, row in batch_df.iterrows():
                with zipfile.ZipFile('images.zip', 'r') as zip_ref:
                    with zip_ref.open(row['image_path']) as file:
                        img = Image.open(file)
                        img = img.resize(img_size)
                        img_array = image_preprocessing(img, **preprocessing_params)
                        img_array = img_to_array(img_array)
                        img_array /= 255.0
                        batch_images.append(img_array)
                        batch_labels.append(label_encoder.transform([row['labels']])[0])
            batch_labels = tf.keras.utils.to_categorical(batch_labels, num_classes=num_classes)
            yield np.array(batch_images), np.array(batch_labels)

img_size = (299, 299)
batch_size = 32
epochs = 10

# Example usage of custom_data_generator with preprocessing
preprocessing_params = {
    'normalize_pixel_vals': True,
    'flattening': True,
    'new_dimensions': (100, 100),
    'sharpening': True,
    'resize_factor': 0.5,
    'to_greyscale': True
}

base_model = InceptionV3(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    custom_data_generator(train_df, batch_size, img_size, preprocessing_params=preprocessing_params),
    steps_per_epoch=len(train_df) // batch_size,
    epochs=epochs,
    validation_data=custom_data_generator(test_df, batch_size, img_size, preprocessing_params=preprocessing_params),
    validation_steps=len(test_df) // batch_size
)

# Save the trained model
model.save('googlenet.h5')