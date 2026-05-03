# 🩺 Skin Disease Classification using CNN

## 📌 Project Overview
This project is a Deep Learning application designed to classify various skin diseases from digital images. It utilizes a **Convolutional Neural Network (CNN)** built with **TensorFlow** and **Keras** to provide automated diagnostic support.
![Model Prediction Output](output_sample.png.png)

## 📂 Dataset Structure
The data is organized into three main directories:
* **Train**: Used for model learning with data augmentation (rotation, shifts, flips).
* **Validate**: Used to tune hyperparameters and monitor for overfitting.
* **Test**: Used for final performance evaluation.

## 🧠 Model Architecture
The model consists of a sequential pipeline:
* **Feature Extraction**: Three `Conv2D` layers with `ReLU` activation and `MaxPooling2D` for downsampling.
* **Global Pooling**: A `GlobalAveragePooling2D` layer to reduce dimensionality.
* **Classification Head**: A `Dense` layer with 128 units, `Dropout(0.5)` for regularization, and a final `Softmax` layer for multi-class classification.

## 🛠️ Technologies Used
* **Python**
* **TensorFlow / Keras**
* **NumPy & Matplotlib**
* **OpenCV / Pillow** (for image processing)

## 🚀 How to Run
1. Ensure you have the `Skindisease_Model.h5` file in your project directory.
2. Run `CNN.py` to predict skin disease from an image path.
3. The output will display the image with its **Prediction Label** and **Confidence Accuracy**.
