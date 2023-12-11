import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from load_images import load_images

# Load images and labels
df = load_images_from_zip('images.zip')
df.head()

# Encode labels
label_encoder = LabelEncoder()
df['labels_encoded'] = label_encoder.fit_transform(df['labels'])
num_classes = len(label_encoder.classes_)

# Split the dataset into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

def custom_data_generator(dataframe, batch_size, img_size, shuffle=True):
    while True:
        if shuffle:
            dataframe = dataframe.sample(frac=1).reset_index(drop=True)
        
        for i in range(0, len(dataframe), batch_size):
            batch_df = dataframe.iloc[i:i + batch_size]
            batch_images = []
            batch_labels = []
            for _, row in batch_df.iterrows():
                img = Image.open(row['image_path'])
                img = img.resize(img_size)
                img_array = img_to_array(img)
                img_array /= 255.0
                if img_array.shape[-1] == 1:
                    img_array = np.concatenate([img_array] * 3, axis=-1)
                batch_images.append(img_array)
                batch_labels.append(label_encoder.transform([row['labels']])[0])
            batch_labels = tf.keras.utils.to_categorical(batch_labels, num_classes=num_classes)
            yield np.array(batch_images), np.array(batch_labels)


img_size = (299, 299)
batch_size = 32
epochs = 5

# Example usage of custom_data_generator with preprocessing
preprocessing_params = {
    'normalize_pixel_vals': True,
    'flattening': True,
    'new_dimensions': (100, 100),
    'sharpening': True,
    'resize_factor': 0.5,
    'to_greyscale': True
}

# Build the model
base_model = InceptionV3(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Train the model
history = model.fit(
    custom_data_generator(train_df, batch_size, img_size, preprocessing_params=preprocessing_params),
    steps_per_epoch=len(train_df) // batch_size,
    epochs=epochs,
    validation_data=custom_data_generator(test_df, batch_size, img_size, preprocessing_params=preprocessing_params),
    validation_steps=len(test_df) // batch_size
)

# Plot training history
def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.show()

# Plot the training history
plot_training_history(history)

# Save the trained model
model.save('googlenet.h5')

# Generate predictions on the test set
test_generator = custom_data_generator(test_df, batch_size, img_size, shuffle=False)
test_steps = len(test_df) // batch_size
test_results = model.evaluate(test_generator, steps=test_steps)

# Print or log the test accuracy
test_accuracy = test_results[1]
print("Test Accuracy:", test_accuracy)

test_generator = custom_data_generator(test_df, batch_size, img_size, shuffle=False)

predictions = model.predict(test_generator, steps=test_steps)

# Convert predictions to labels
predicted_labels = np.argmax(predictions, axis=1)
true_labels = test_df['labels_encoded'].values

# Compute precision, recall, and other metrics
precision = precision_score(true_labels, predicted_labels, average='weighted')
recall = recall_score(true_labels, predicted_labels, average='weighted')

# Print or log precision and recall
print("Precision:", precision)
print("Recall:", recall)

# Print or log the detailed classification report
classification_report_str = classification_report(true_labels, predicted_labels, target_names=label_encoder.classes_)
print("Classification Report:")
print(classification_report_str)