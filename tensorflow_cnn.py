from load_images import load_images_from_zip
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from PIL import Image
import zipfile
from image_preprocessing import image_preprocessing
import os
from tensorflow.keras.models import saved_model

# Load image data
dataset_path = 'images.zip'
df = load_images_from_zip(dataset_path)

# Encode labels
label_encoder = LabelEncoder()
df['labels_encoded'] = label_encoder.fit_transform(df['labels'])
num_classes = len(label_encoder.classes_)

# Split data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Image preprocessing function
def image_preprocessing(img, **preprocessing_params):
    img = img.resize(preprocessing_params['img_size'])
    img_array = img_to_array(img)
    img_array /= 255.0  # Normalize pixel values
    return img_array

# Custom data generator
def custom_data_generator(dataframe, batch_size, img_size, preprocessing_params):
    datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=preprocessing_params.get('validation_split', 0.0),
    )
    
    while True:
        for i in range(0, len(dataframe), batch_size):
            batch_df = dataframe.iloc[i:i + batch_size]
            batch_images = []
            batch_labels = []
            
            for index, row in batch_df.iterrows():
                with zipfile.ZipFile('images.zip', 'r') as zip_ref:
                    with zip_ref.open(row['image_path']) as file:
                        img = Image.open(file)
                        img_array = image_preprocessing(img, **preprocessing_params)
                        batch_images.append(img_array)
                        batch_labels.append(label_encoder.transform([row['labels']])[0])

            # Apply data augmentation and normalization
            batch_images = np.array(batch_images)
            batch_labels = tf.keras.utils.to_categorical(batch_labels, num_classes=num_classes)
            batch = datagen.flow(batch_images, batch_labels, batch_size=batch_size, shuffle=False).next()

            yield batch[0], batch[1]

img_size = (299, 299)
batch_size = 32
epochs = 10

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(299, 299, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    custom_data_generator(train_df, batch_size, img_size, preprocessing_params={'img_size': img_size}),
    steps_per_epoch=len(train_df) // batch_size,
    epochs=epochs,
    validation_data=custom_data_generator(test_df, batch_size, img_size, preprocessing_params={'img_size': img_size}),
    validation_steps=len(test_df) // batch_size
)

# Save the trained model
model.save('tensorflow_cnn.h5')