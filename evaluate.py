# evaluate.py

from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np

# 1. Load test data
(_, _), (X_test, y_test) = mnist.load_data()

# 2. Preprocess test images
X_test = X_test.astype("float32") / 255.0
X_test = np.expand_dims(X_test, axis=-1)  # shape: (10000, 28, 28, 1)
y_test = to_categorical(y_test, 10)

# 3. Load the trained model
import os
model_path = os.path.join(os.getcwd(), "saved_models", "model.h5")
model = load_model(model_path)


# 4. Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=2)

# 5. Print results
print(f"\n✅ Test Accuracy: {accuracy * 100:.2f}%")
print(f"📉 Test Loss: {loss:.4f}")
