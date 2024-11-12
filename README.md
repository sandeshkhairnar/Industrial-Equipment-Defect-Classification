# 📊 Industrial Equipment Defect Classification

## 📌 Overview
This project focuses on developing a deep learning model to classify images of industrial equipment into two categories: **defective** and **non-defective**. Leveraging a pre-trained **ResNet50** model for efficient feature extraction, the solution includes additional fully connected layers for binary classification. This approach aids in detecting defects in industrial equipment, contributing to quality control and predictive maintenance.

---

## 🗂️ Project Structure
- **Dataset**:
  - Organized into `train` and `test` directories with subfolders for 'defective' and 'non-defective' classes.
  - Stored in Google Drive and linked to the Colab environment.
- **Model Architecture**:
  - **Base Model**: **ResNet50** (pre-trained on ImageNet).
  - **Additional Layers**:
    - Global Average Pooling
    - Dense layers with ReLU activation
    - Dropout for regularization
    - Sigmoid activation for binary classification
- **Evaluation Metrics**: Accuracy, Precision, Recall, and Loss are used to evaluate model performance.

---

## 📁 Key Files
- **`model_training.ipynb`**: Jupyter notebook containing the full implementation, including data preprocessing, model training, and evaluation.
- **`README.md`**: Provides an overview of the project, setup instructions, and insights.

---

## 🔧 Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/sandeshkhairnar/Industrial-Equipment-Defect-Classification.git
cd defect-classification
---
## 2.  Install Dependencies
   Make sure you have Python and pip installed, then run:
```bash
pip install tensorflow numpy matplotlib pandas

---
## 3. Prepare the Dataset
- Download and extract the dataset.
- Upload it to your Google Drive.
- Mount Google Drive in Google Colab:
```bash
from google.colab import drive
drive.mount('/content/drive')

-Ensure the dataset path is correctly set in the notebook.

---

## 4. Run the Jupyter Notebook
-Open model_training.ipynb in Google Colab.
-Adjust hyperparameters as needed.
-Run the notebook cells sequentially to train and evaluate the model.

---
## 🧑‍💻 Model Training & Compilation

- a pre-trained ResNet50 model is used as a feature extractor for classifying images of industrial equipment into two categories: defective or non-defective. The base model (ResNet50) is frozen to prevent retraining, and a series of additional layers (global average pooling, dense layers, dropout, and sigmoid output) are added to adapt the network to the binary classification task. The model is then compiled and ready for training on the target dataset.

---

## 📊 Model Evaluation

---
## 📈 Results & Insights

-The model showed strong performance on the training data but exhibited signs of overfitting on the validation data.
-Fine-tuning the learning rate, dropout rate, and adding data augmentation could enhance generalization.
