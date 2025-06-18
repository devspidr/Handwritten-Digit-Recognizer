import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

st.set_page_config(page_title="Digit Recognizer", page_icon="✍️")

# Load model
model = load_model("saved_models/model.h5")

st.title("🧠 Handwritten Digit Recognizer")
st.write("Draw a digit (0–9) and the model will predict it!")

# Upload or draw
uploaded_file = st.file_uploader("Upload a digit image (28x28 or larger)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded Digit", use_column_width=True)

    # Preprocess
    image = ImageOps.invert(image)
    image = image.resize((28, 28))
    image = np.array(image)
    _, image = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    st.success(f"🔢 Predicted Digit: {predicted_digit}")
    st.info(f"📊 Confidence: {confidence:.2f}%")
