A deep learning project focused on classifying digits from natural street-level images using both Artificial Neural Networks (ANN) and Convolutional Neural Networks (CNN). This notebook explores data preparation, modeling, and performance evaluation across several architectures to identify the best approach.

📌 Project Overview
Goal:
To classify digits (0–9) extracted from the Street View House Numbers (SVHN) dataset using both fully connected neural networks (ANNs) and convolutional neural networks (CNNs), comparing model architectures and evaluating performance on a held-out test set.

🧠 Models Trained
🔹 ANN Model 1
Simple architecture: 64 → 32 → 10 (softmax)

Accuracy: ~69%

Minimal overfitting, but overall weak performance for image classification

🔹 ANN Model 2
Deeper network with more hidden layers and added dropout

Accuracy: ~72%

Improved generalization but still limited by the nature of dense networks on image data

🔹 CNN Model 1
2 Conv2D layers + MaxPooling + Dense

Accuracy: ~91%

High train accuracy but signs of overfitting on validation set

🔹 CNN Model 2 ✅ Best Performer
Deeper CNN with BatchNormalization and Dropout

Accuracy: ~95% (train), ~91% (validation)

Balanced performance with minimal overfitting and excellent generalization

📂 Dataset
SVHN dataset provided in .h5 format (subset used to reduce training time)

Pre-split into X_train, X_val, and X_test

Images are grayscale and of size 32x32

⚙️ Workflow
Data Loading – Load .h5 formatted dataset into memory

Exploration – Visualize samples and inspect class distribution

Preprocessing – Normalize pixel values, reshape inputs, one-hot encode labels

Model Building – Define ANN and CNN architectures using Keras

Training & Evaluation – Fit each model and visualize learning curves

Prediction – Generate predictions on test set and evaluate using confusion matrix and classification report

Comparison & Conclusion – Assess model generalization and recommend best architecture

📊 Key Observations
ANN
Easy to implement and fast to train

Lacks spatial awareness → limited classification power for image data

Model complexity improves performance marginally but saturates quickly

CNN
Significantly outperforms ANN due to spatial feature extraction

Adding BatchNormalization and Dropout greatly reduces overfitting

Best model achieves high performance on both training and validation sets

🏁 Final Verdict
✅ CNN Model 2 is the recommended model for digit classification in this project due to its high accuracy, generalization capability, and robustness across all classes.

💻 Tech Stack
Python 3.x

TensorFlow / Keras

NumPy, Pandas, Seaborn, Matplotlib

Scikit-learn

Google Colab environment

📸 Sample Outputs
<details> <summary>📷 Sample Training Image</summary> <img src="path-to-sample-image.png" width="400"/> </details> <details> <summary>📈 Training vs Validation Accuracy</summary>
Include training curve plots here if you'd like to embed them as images.

</details>
🚀 Run This Project
bash
Copy
# Clone the repo
git clone https://github.com/yourusername/svhn-digit-recognition

# Open the Jupyter Notebook
# Run cells sequentially in Colab or Jupyter environment
📌 Future Enhancements
Implement early stopping & model checkpointing

Try pretrained CNNs like VGG or ResNet for transfer learning

Explore other datasets or multi-digit house number recognition

Deploy as a web-based digit recognition app
