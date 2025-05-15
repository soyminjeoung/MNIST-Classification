# MNIST Classification using Deep Neural Network

This project applies a **single-layer deep neural network (DNN)** to classify handwritten digits from the **MNIST dataset**. The focus is on experimenting with different number of neurons to observe their effects on model performance.

## 🧠 Objective

To build and evaluate a basic deep learning model for digit recognition, while learning how a single layer activates neurons and a different number of features influences classification accuracy.

---

## 📊 Dataset

- **Dataset:** [MNIST Handwritten Digits](http://yann.lecun.com/exdb/mnist/)
- **Description:** 60,000 training images and 10,000 test images of handwritten digits (0–9), each image is 28x28 pixels in grayscale.

---

## 🧪 Experiments

The following model configurations were tested:

- **Hidden units:** 1, 2, 8, 16, 32, 64, 128, 256
- **Feature Selection:** PCA, Random Forest Classifier
- **Activation functions:** ReLU, Softmax 
- **Optimizers:** root mean squared propagation (rmsprop)
- **Batch sizes:** 32
- **Epochs:** Up to 50 (with early stopping)

Each configuration was evaluated based on validation accuracy and test accuracy.

---

## 🧰 Tools & Libraries

- Python 3.13.3
- TensorFlow / Keras  
- NumPy, Matplotlib for visualization  

---

## 📈 Results Summary

- Best performance achieved with:
  - **256 hidden neurons**
  - **ReLU activation**
  - **rmsprop optimizer**
  - **Batch size of 32**
  - **Full 784 Features**

> Final test accuracy: **0.9699**

Visualizations include training/validation accuracy plots and confusion matrix of final predictions.

---

---

## 🚀 How to Run

You can run the notebook directly in **[Google Colab](https://colab.research.google.com/)** or locally:

1. Clone the repo  
   `git clone https://github.com/yourusername/mnist-dnn-hyperparameter.git`

2. Open `mnist_dnn.ipynb` in Jupyter or Colab

3. Run all cells to train the model

---

## 📌 Learnings

- Importance of tuning hidden units and optimizers for faster convergence
- Impact of activation functions on performance and training stability
- How batch size affects noise vs. convergence speed

---

## 🧠 Author

**Minjeoung Neev**  
Graduate student passionate about deep learning & data science  
📧 laminjon@gmail.com

---

## 📜 License

This project is under the MIT License. Feel free to fork and experiment!

