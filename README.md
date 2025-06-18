# 🧠 Handwritten Digit Recognizer (MNIST - CNN)

This is a simple deep learning project that uses a Convolutional Neural Network (CNN) to recognize handwritten digits (0–9) using the MNIST dataset.

## 📂 Project Structure

- `train.py` - Trains the CNN on MNIST
- `evaluate.py` - Evaluates the trained model
- `visualize_predictions.py` - Displays predictions on test images
- `predict.py` - Predicts on your own digit image (`own_digit.png`)
- `saved_models/model.h5` - Saved trained model
- `own_digit.png` - Sample image for prediction

## 🖼️ Sample Output

🔢 Predicted Digit: 3
📊 Confidence: 93.55%



## ▶️ How to Run

1. Clone/download this repo
2. Install dependencies:
pip install -r requirements.txt
3. Run training:
python train.py
4. Predict your own image:
python predict.py

## 🧠 Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- OpenCV
- Matplotlib
