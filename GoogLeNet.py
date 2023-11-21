import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from load_images import load_images_from_zip
import zipfile

# Load images and labels
df = load_images_from_zip('images.zip')

# Encode labels
label_encoder = LabelEncoder()
df['labels_encoded'] = label_encoder.fit_transform(df['labels'])
num_classes = len(label_encoder.classes_)

# Split the dataset into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

def custom_data_generator(dataframe, batch_size, img_size):
    while True:
        for i in range(0, len(dataframe), batch_size):
            batch_df = dataframe.iloc[i:i+batch_size]
            batch_images = []
            batch_labels = []
            for _, row in batch_df.iterrows():
                with zipfile.ZipFile('images.zip', 'r') as zip_ref:
                    with zip_ref.open(row['image_path']) as file:
                        img = Image.open(file)
                        img = img.resize(img_size)
                        img_array = img_to_array(img)
                        img_array /= 255.0
                        batch_images.append(img_array)
                        batch_labels.append(label_encoder.transform([row['labels']])[0])
            batch_labels = tf.keras.utils.to_categorical(batch_labels, num_classes=num_classes)
            yield np.array(batch_images), np.array(batch_labels)

img_size = (299, 299)
batch_size = 32
epochs = 10

base_model = InceptionV3(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Train the model
history = model.fit(
    custom_data_generator(train_df, batch_size, img_size),
    steps_per_epoch=len(train_df) // batch_size,
    epochs=epochs,
    validation_data=custom_data_generator(test_df, batch_size, img_size),
    validation_steps=len(test_df) // batch_size
)

# Save the trained model
model.save('googlenet.h5')

# Evaluate the model on the test set
test_generator = custom_data_generator(test_df, batch_size, img_size)
test_steps = len(test_df) // batch_size
test_predictions = model.predict(test_generator, steps=test_steps)

# Convert predictions to class labels
predicted_labels = np.argmax(test_predictions, axis=1)
true_labels = np.concatenate([np.argmax(y_true, axis=1) for _, y_true in test_generator], axis=0)

# Calculate accuracy, precision, and recall
accuracy = accuracy_score(true_labels, predicted_labels)
report = classification_report(true_labels, predicted_labels, target_names=label_encoder.classes_)

# Print or log the results
print("Test Accuracy:", accuracy)
print("Classification Report:\n", report)