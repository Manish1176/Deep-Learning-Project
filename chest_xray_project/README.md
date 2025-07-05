# Chest X-ray Classification Using CNN

This project is a chest X-ray image classification pipeline that distinguishes between **Normal** and **Abnormal** chest X-rays using a custom-built Convolutional Neural Network (CNN). The data comes from a large dataset that originally contains over 111,000 records.

---

## 🧠 Problem Statement

To build a predictive model that classifies grayscale chest X-ray images into:
- **Label 0**: Normal (No Finding)
- **Label 1**: Abnormal (Any disease present)

---

## 📁 Dataset Overview

- **CSV File**: `ground_truth.csv` with labels, age, gender, etc.
- **Images Folder**: `xray_images/` contains actual image files
- Only **3681 images** matched with entries in the CSV (filtered using file existence)

---

## ✅ Approach & Pipeline

### 1. **Preprocessing**
- Grayscale conversion using PIL
- Resize to 224×224
- Normalize to [0, 1]
- Reshape to (224, 224, 1)

### 2. **Model Architecture**
- CNN with 3 Conv-Pool blocks
- Flatten → Dense(128) + Dropout → Dense(2) with softmax

### 3. **Training Details**
- 10 epochs
- Adam optimizer
- Batch size = 32
- Validation split = 10%

---

## 📉 Evaluation

- **Test Accuracy**: ~61.19%
- **Precision/Recall**:
  - Label 0 (Normal): Precision = 0.63, Recall = 0.82
  - Label 1 (Abnormal): Precision = 0.57, Recall = 0.32

---

## 🔍 Challenges Faced

- Huge dataset (111k+), but only ~3681 usable images after filtering
- Imbalance in classes → model favors "Normal" predictions
- Low recall on "Abnormal" class (missed many disease-positive cases)

---

## 🧪 Testing & Inference

- A predictive system accepts a path to a chest X-ray image
- Preprocesses and reshapes the image
- Displays the image and prints prediction results

---

## 🔧 How to Improve the Model

### Data-Level Improvements:
- Use more data (process all 111k images)
- Data augmentation (rotation, shift, zoom, flip)
- Oversample minority class (label 1)

### Model-Level Improvements:
- Use transfer learning (ResNet50, MobileNet)
- Apply early stopping and learning rate scheduler
- Add class weights during training to address imbalance

### Evaluation Enhancements:
- Include confusion matrix and ROC-AUC
- Calibrate threshold for better recall vs precision balance

---

## 💻 Requirements

- Python 3.x
- TensorFlow / Keras
- OpenCV
- NumPy, Pandas, Matplotlib
- PIL (Pillow)

---

## 📌 Conclusion

This end-to-end project demonstrates the development and evaluation of a CNN-based binary classifier for chest X-ray diagnosis. With more data, augmentation, and class balancing, model performance (especially recall) can be significantly improved.

> Developed by Manish using Google Colab and TensorFlow.

