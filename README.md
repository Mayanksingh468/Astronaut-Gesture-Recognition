# 🚀 Astronaut Hand Gesture Recognition using Transfer Learning

## 📌 Project Overview

Hand gesture recognition can provide an intuitive communication interface in environments where traditional input devices are impractical. Inspired by astronaut communication systems, this project investigates the effectiveness of transfer learning architectures for recognizing static hand gestures.

Three popular CNN architectures were evaluated:

* ResNet50
* MobileNetV2
* EfficientNetB0

The models were trained and fine-tuned on the ASL Alphabet Dataset containing approximately 87,000 images across 29 gesture classes.

---

## 🎯 Objectives

* Compare the performance of modern transfer learning architectures.
* Evaluate their effectiveness on large-scale hand gesture classification.
* Fine-tune the best-performing model.
* Deploy the final model as an interactive web application.

---

## 📂 Dataset

**ASL Alphabet Dataset**

Dataset Characteristics:

* ~87,000 RGB Images
* 29 Gesture Classes
* Static Hand Gestures
* Image Resolution: 224 × 224
* Balanced Class Distribution

---

## ⚙️ Methodology

### 1. Data Preprocessing

* Image resizing
* Normalization
* Train-validation-test split

### 2. Data Augmentation

* Rotation
* Horizontal shifting
* Zooming
* Flipping

### 3. Transfer Learning

Pre-trained ImageNet weights were used to accelerate convergence and improve generalization.

### 4. Fine-Tuning

The top layers of each architecture were fine-tuned on the gesture dataset.

### 5. Evaluation

Models were compared using:

* Validation Accuracy
* Test Accuracy
* Confusion Matrix Analysis

---

## 📊 Model Performance

| Model          | Validation Accuracy | Test Accuracy |
| -------------- | ------------------- | ------------- |
| ResNet50       | 80.98%              | 80.11%        |
| MobileNetV2    | 75.76%              | 75.48%        |
| EfficientNetB0 | **90.81%**          | **89.87%**    |

---

## 🏆 Best Performing Model

### EfficientNetB0

Results achieved:

* Validation Accuracy: **90.81%**
* Test Accuracy: **89.87%**
* Highest overall performance among all evaluated architectures

EfficientNetB0 demonstrated superior feature extraction capability while maintaining computational efficiency, making it the final model selected for deployment.

---

## 💻 Deployed Application

The best-performing EfficientNetB0 model was deployed using Streamlit to provide real-time gesture prediction.

### Features

* Upload hand gesture images
* Real-time predictions
* Confidence score display
* Top-3 prediction probabilities
* Cloud deployment using Streamlit

---

## 📸 Application Preview

### Home Page

(Add Screenshot Here)

### Prediction Example

(Add Screenshot Here)

---

## 📈 Confusion Matrix

![Confusion Matrix](images/efficientnet_confusion_matrix.png)

---

## 🛠️ Technologies Used

### Machine Learning & Deep Learning

* TensorFlow
* Keras
* Transfer Learning

### Data Processing

* NumPy
* Pandas

### Visualization

* Matplotlib
* Seaborn

### Deployment

* Streamlit

### Version Control

* Git
* GitHub

---

## 📁 Repository Structure

```text
Astronaut-Gesture-Recognition/
│
├── app/
│   ├── app.py
│   └── efficientnetb0_finetuned.keras
│
├── notebooks/
│   └── ASL_Gesture_Recognition.ipynb
│
├── images/
│   └── efficientnet_confusion_matrix.png
│
├── requirements.txt
├── runtime.txt
└── README.md
```

---

## 🔬 Key Insights

* Transfer learning significantly reduced training time.
* EfficientNetB0 achieved substantially higher accuracy than ResNet50 and MobileNetV2.
* Fine-tuning improved model generalization on unseen gesture images.
* Deployment demonstrated the practical applicability of the trained model.

---

## 🚀 Future Improvements

* Grad-CAM Visualizations
* Real-Time Webcam Gesture Recognition
* Dynamic Gesture Recognition
* Model Quantization for Edge Deployment
* Support for Custom Gesture Datasets

---

## 👨‍💻 Author

Mayank Singh

GitHub:
https://github.com/Mayanksingh468

Project Repository:
https://github.com/Mayanksingh468/Astronaut-Gesture-Recognition
