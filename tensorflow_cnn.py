from sklearn.metrics import classification_report, precision_score, recall_score
from load_images import load_images
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from PIL import Image

# Load image data
df = load_images('images')

# Encode labels
label_encoder = LabelEncoder()
df['labels_encoded'] = label_encoder.fit_transform(df['labels'])
num_classes = len(label_encoder.classes_)

# Split data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

def custom_data_generator(dataframe, batch_size, img_size, shuffle=True):
    datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    while True:
        if shuffle:
            dataframe = dataframe.sample(frac=1).reset_index(drop=True)  # Shuffle the data

        for i in range(0, len(dataframe), batch_size):
            batch_df = dataframe.iloc[i:i+batch_size]
            batch_images = []
            batch_labels = []
            for _, row in batch_df.iterrows():
                img = Image.open(row['image_path'])
                img = img.resize(img_size)
                img_array = img_to_array(img)
                if img_array.shape[-1] == 1:
                    img_array = np.concatenate([img_array] * 3, axis=-1)
                batch_images.append(img_array)
                batch_labels.append(label_encoder.transform([row['labels']])[0])
            
            # Apply data augmentation and normalization
            batch_images = datagen.flow(np.array(batch_images), shuffle=False, batch_size=batch_size).next()
            
            batch_labels = tf.keras.utils.to_categorical(batch_labels, num_classes=num_classes)
            yield batch_images, batch_labels

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
    custom_data_generator(train_df, batch_size, img_size),
    steps_per_epoch=len(train_df) // batch_size,
    epochs=epochs,
    validation_data=custom_data_generator(test_df, batch_size, img_size),
    validation_steps=len(test_df) // batch_size
)

# Save the trained model
model.save('tensorflow_cnn.keras')

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
