# 🔢 SVHN Digit Recognition – CNN & ANN Comparison

Welcome to my deep learning project on digit classification using the **Street View House Numbers (SVHN)** dataset. This notebook-based analysis explores how different neural network architectures—Artificial Neural Networks (ANNs) and Convolutional Neural Networks (CNNs)—perform on digit recognition from real-world, street-level images.

> 📍 **Tech Stack**: TensorFlow · Keras · Python · NumPy · Matplotlib · Seaborn · Google Colab  
> 🎯 **Purpose**: Image classification | Model comparison | Hands-on CNN/ANN architecture exploration

---

## 📊 Project Summary

Using a pre-split `.h5` version of the SVHN dataset, I built, trained, and evaluated multiple neural network models to identify the most effective structure for digit classification:

### 🧠 Models Trained

#### 🔹 ANN Model 1  
- **Architecture**: Dense(64) → Dense(32) → Dense(10, softmax)  
- **Accuracy**: ~69%  
- Fast to train, low performance on image data, minimal overfitting

#### 🔹 ANN Model 2  
- **Deeper architecture with Dropout layers**  
- **Accuracy**: ~72%  
- Better generalization but still limited due to no spatial context

#### 🔹 CNN Model 1  
- **Architecture**: Conv2D → Conv2D → MaxPooling → Dense  
- **Accuracy**: ~91%  
- Strong performance, but overfitting noted on validation set

#### 🔹 CNN Model 2 ✅ *Best Performer*  
- **Architecture**: Deeper CNN + BatchNormalization + Dropout  
- **Accuracy**: ~95% (train), ~91% (validation)  
- Best generalization and robustness across all models

---

## 📂 Dataset

- Source: [SVHN Dataset](http://ufldl.stanford.edu/housenumbers/) (cropped digit format)  
- Format: `.h5` with `X_train`, `X_val`, `X_test`  
- Image size: 32x32 pixels, grayscale  
- Target labels: One-hot encoded (0–9)

---

## ⚙️ Workflow Breakdown

### 🧹 Data Prep  
- Normalization, reshaping, and one-hot encoding  
- ANN inputs reshaped to 1D (1024 features), CNNs used full 4D tensors

### 🧱 Model Architecture & Training  
- Sequential models built with TensorFlow/Keras  
- Trained with early observations and performance visualizations  
- Learning rate, dropout, and batch size tuned over iterations

### 📈 Evaluation  
- Accuracy trends visualized  
- Test predictions evaluated using `classification_report` and `confusion_matrix`  
- Observed misclassifications, especially between visually similar digits (e.g., 3 & 5)

---

## 🧠 Key Learnings & Observations

### ANN Takeaways  
- ANN models are simple and fast but underperform for image tasks  
- Accuracy improvements diminish with additional complexity  

### CNN Takeaways  
- Outperforms ANN significantly due to spatial feature learning  
- BatchNormalization and Dropout improve generalization  
- CNN Model 2 demonstrates near state-of-the-art performance on this subset

---

## 🏁 Final Verdict

✅ **CNN Model 2 is the recommended architecture**  
It delivers the highest validation accuracy and lowest overfitting, making it the most reliable model in this project.

---

## 🚀 Run This Project

```bash
# Clone the repo
git clone https://github.com/yourusername/svhn-digit-recognition

# Open the .ipynb file in Colab or Jupyter
# Run all cells to see model training and results
