# train.py

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from models.cnn_model import create_model
import numpy as np

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize to [0,1] and reshape to (28, 28, 1)
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0
X_train = np.expand_dims(X_train, axis=-1)  # shape: (60000, 28, 28, 1)
X_test = np.expand_dims(X_test, axis=-1)    # shape: (10000, 28, 28, 1)

# One-hot encode the labels (e.g., 3 → [0,0,0,1,0,0,0,0,0,0])
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Build CNN model
model = create_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

# Save the model
model.save("saved_models/model.h5")
print("✅ Model trained and saved at: saved_models/model.h5")
