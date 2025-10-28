# 🌦️ Multi-Class Weather Classification using PyTorch

## 📘 Overview  
This project uses a **Convolutional Neural Network (CNN)** built with **PyTorch** to classify weather images into four categories — **Shine, Cloudy, Rain,** and **Sunrise**.  
The model learns weather features from image textures, color tones, and lighting conditions, achieving high accuracy on the validation set.

---

## 📊 Dataset  
**Dataset:** [Multi-class Weather Dataset](https://www.kaggle.com/datasets/pratik2901/multiclass-weather-dataset)

- **Classes:**  
  - 🌤️ Shine  
  - ☁️ Cloudy  
  - 🌧️ Rain  
  - 🌅 Sunrise  
- Each folder contains real-world weather images.  
- The dataset was divided into **training** and **validation** sets using `train_test_split`.  

---

## ⚙️ Tech Stack
- **Language:** Python  
- **Framework:** PyTorch  
- **Libraries:** torchvision, numpy, pandas, matplotlib, seaborn  
- **Environment:** Kaggle / Jupyter Notebook  

---

## 🧠 Model Architecture
A **custom CNN** consisting of:
- Convolutional + BatchNorm + ReLU + MaxPooling layers  
- Fully connected layers with Dropout  
- Softmax activation for 4-class output  

Optionally, transfer learning models (ResNet18/34) can be used for higher accuracy.

---

## 🚀 Features
✅ Data visualization & augmentation  
✅ Training with live progress (tqdm)  
✅ Validation accuracy tracking  
✅ Plotting loss & accuracy curves  
✅ Predicting new weather images with confidence scores  
✅ GPU/CPU compatible  

---

## 📈 Results
- **Validation Accuracy:** ~99%  
- **Training Time:** ~7s per epoch (GPU)  
- Stable convergence and minimal loss  

---

## 🖼️ Example Prediction

```python
# Example call:

img_path = "/kaggle/input/multiclass-weather-dataset/Multi-class Weather Dataset/Sunrise/sunrise10.jpg"
label, conf = predict_image(img_path, model, transform, classes, device)
print(f"Predicted Weather: {label}")
print(f"Confidence: {conf:.2f}%")

<img width="515" height="456" alt="Screenshot from 2025-10-28 12-53-48" src="https://github.com/user-attachments/assets/498628fe-cf66-4e4c-9287-eecafd71d637" />
