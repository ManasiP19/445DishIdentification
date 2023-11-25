from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'test_df' is your test DataFrame
# Assuming 'model' is your GoogleNet model
# Assuming 'model_tf_cnn' is your TensorFlow CNN model

# GoogleNet model evaluation
test_generator_googleNet = custom_data_generator(test_df, batch_size, img_size)
test_steps_googleNet = len(test_df) // batch_size
test_predictions_googleNet = model.predict(test_generator_googleNet, steps=test_steps_googleNet)

predicted_labels_googleNet = np.argmax(test_predictions_googleNet, axis=1)
true_labels_googleNet = np.concatenate([np.argmax(y_true, axis=1) for _, y_true in test_generator_googleNet], axis=0)

# TensorFlow CNN model evaluation
test_generator_tf_cnn = custom_data_generator(test_df, batch_size, img_size)
test_steps_tf_cnn = len(test_df) // batch_size
test_predictions_tf_cnn = model_tf_cnn.predict(test_generator_tf_cnn, steps=test_steps_tf_cnn)

predicted_labels_tf_cnn = np.argmax(test_predictions_tf_cnn, axis=1)
true_labels_tf_cnn = np.concatenate([np.argmax(y_true, axis=1) for _, y_true in test_generator_tf_cnn], axis=0)

# Plot confusion matrix for GoogleNet
cm_googleNet = confusion_matrix(true_labels_googleNet, predicted_labels_googleNet)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_googleNet, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix - GoogleNet')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Plot confusion matrix for TensorFlow CNN
cm_tf_cnn = confusion_matrix(true_labels_tf_cnn, predicted_labels_tf_cnn)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_tf_cnn, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix - TensorFlow CNN')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
