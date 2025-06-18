# visualize_predictions.py

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist

# Load model
from tensorflow.keras.models import load_model
import os

# Get absolute path to model file
base_dir = os.path.dirname(os.path.dirname(__file__))  # go up one directory
model_path = os.path.join(base_dir, "saved_models", "model.h5")
model = load_model(model_path)


# Load test data
(_, _), (X_test, y_test) = mnist.load_data()

# Preprocess test images
X_test_normalized = X_test.astype("float32") / 255.0
X_test_input = np.expand_dims(X_test_normalized, axis=-1)

# Predict classes for first 10 test images
predictions = model.predict(X_test_input[:10])
predicted_labels = np.argmax(predictions, axis=1)

# Plot predictions
plt.figure(figsize=(12, 5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_test[i], cmap="gray")
    plt.title(f"Pred: {predicted_labels[i]} | True: {y_test[i]}")
    plt.axis("off")
plt.tight_layout()
plt.show()
