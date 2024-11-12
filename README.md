# Industrial Equipment Defect Classification

## Overview
This project focuses on developing a deep learning model to classify images of industrial equipment into two categories: **defective** and **non-defective**. Leveraging a pre-trained **ResNet50** model for efficient feature extraction, the solution includes additional fully connected layers for binary classification. This approach aids in detecting defects in industrial equipment, contributing to quality control and predictive maintenance.

## Project Structure
- **Dataset**: Images are organized into training and testing directories with subfolders for 'defective' and 'non-defective' classes.
- **Model Architecture**:
  - **Base Model**: ResNet50 (pre-trained on ImageNet).
  - **Additional Layers**: Global Average Pooling, Dense layers, and Dropout for regularization.
- **Evaluation Metrics**: Accuracy, Precision, Recall, and Loss are used to evaluate model performance.

## Key Files
- `model_training.ipynb`: Jupyter notebook containing the full implementation, including data preprocessing, model training, and evaluation.
- **Dataset Path**: Images are stored in Google Drive and linked to the Colab environment.
- `README.md`: Provides an overview of the project, setup instructions, and insights.
