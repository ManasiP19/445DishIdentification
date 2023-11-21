import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model

# Load the models
model1 = load_model('googlenet.h5')
model2 = load_model('tensorflow_cnn.h5')

# Load the test data
test_df = ...  # Load your test data here

# Evaluate models on the test set
test_generator1 = custom_data_generator(test_df, batch_size, img_size, preprocessing_params={'img_size': img_size})
test_generator2 = custom_data_generator(test_df, batch_size, img_size, preprocessing_params={'img_size': img_size})

test_steps = len(test_df) // batch_size

# Model 1 evaluation
test_results1 = model1.evaluate(test_generator1, steps=test_steps)
predictions1 = model1.predict(test_generator1, steps=test_steps)
predicted_labels1 = np.argmax(predictions1, axis=1)

# Model 2 evaluation
test_results2 = model2.evaluate(test_generator2, steps=test_steps)
predictions2 = model2.predict(test_generator2, steps=test_steps)
predicted_labels2 = np.argmax(predictions2, axis=1)

# Plot Learning Curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)

plt.plot(history1.history['accuracy'], label='Model 1 Train Accuracy')
plt.plot(history1.history['val_accuracy'], label='Model 1 Validation Accuracy')
plt.plot(history2.history['accuracy'], label='Model 2 Train Accuracy')
plt.plot(history2.history['val_accuracy'], label='Model 2 Validation Accuracy')

plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)

plt.plot(history1.history['loss'], label='Model 1 Train Loss')
plt.plot(history1.history['val_loss'], label='Model 1 Validation Loss')
plt.plot(history2.history['loss'], label='Model 2 Train Loss')
plt.plot(history2.history['val_loss'], label='Model 2 Validation Loss')

plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Plot Confusion Matrices
def plot_confusion_matrix(model_name, predicted_labels, true_labels, class_names):
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

plot_confusion_matrix('Model 1', predicted_labels1, true_labels, label_encoder.classes_)
plot_confusion_matrix('Model 2', predicted_labels2, true_labels, label_encoder.classes_)

# Compare Metrics
def print_classification_report(model_name, predicted_labels, true_labels):
    report = classification_report(true_labels, predicted_labels, target_names=label_encoder.classes_)
    print(f'Classification Report - {model_name}:\n{report}')

print_classification_report('Model 1', predicted_labels1, true_labels)
print_classification_report('Model 2', predicted_labels2, true_labels)
