import numpy as np
import matplotlib.pyplot as plt

# Assuming you have these defined already
epochs = 10
batch_size = 32

# Lists to store training metrics
training_accuracy = [0.0301, 0.1237, 0.2188, 0.3075, 0.3895, 0.4557, 0.5109, 0.5558, 0.5899, 0.6235]
validation_accuracy = [0.0312, 0.0967, 0.2265, 0.2466, 0.3465, 0.3261, 0.4432, 0.3905, 0.3898, 0.4734]
training_loss = [4.3992, 3.66666, 3.1315, 2.7067, 2.3464, 2.0684, 1.8511, 1.6639, 1.15144, 1.3804]
validation_loss = [5.6400, 4.4275, 3.2297, 3.1621, 2.6679, 2.8111, 2.1894, 2.6870, 2.6585, 2.0912]

# Plotting
epochs_range = range(1, epochs + 1)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, training_accuracy, label='Training Accuracy')
plt.plot(epochs_range, validation_accuracy, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, training_loss, label='Training Loss')
plt.plot(epochs_range, validation_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.show()
