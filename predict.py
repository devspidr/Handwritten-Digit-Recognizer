# predict.py

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt

# === Step 1: Load trained model ===
model = load_model("saved_models/model.h5")

# === Step 2: Load and preprocess your own image ===
img_path = "own_digit.png"  # Make sure this file exists in the project root

# Load image in grayscale
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"⚠️ Image not found at: {img_path}")

# Resize while keeping aspect ratio (if needed)
img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

# Invert colors (MNIST = white digit on black background)
img = cv2.bitwise_not(img)

# Apply thresholding to sharpen the digit
_, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

# Normalize and reshape
img = img.astype("float32") / 255.0
img_input = np.expand_dims(img, axis=-1)
img_input = np.expand_dims(img_input, axis=0)

# Optional: Preview preprocessed image
plt.imshow(img, cmap="gray")
plt.title("Preprocessed Digit")
plt.axis("off")
plt.show(block=False)
plt.pause(3)   # display for 3 seconds
plt.close()

# === Step 3: Predict ===
pred = model.predict(img_input)
digit = np.argmax(pred)
confidence = np.max(pred) * 100

# === Step 4: Display result ===
print(f"\n🔢 Predicted Digit: {digit}")
print(f"📊 Confidence: {confidence:.2f}%")
