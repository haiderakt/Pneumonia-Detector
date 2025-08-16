**A deep learning–based chest X-ray classifier to distinguish between NORMAL and PNEUMONIA cases using transfer learning with ResNet18 and an interactive Gradio demo.**

**Pneumonia Detector**
A deep learning–based chest X-ray classifier to distinguish between NORMAL and PNEUMONIA cases using transfer learning with ResNet18 and an interactive Gradio demo.

**Project Overview**
Pneumonia is a significant global health concern, and timely and accurate diagnosis is crucial for effective treatment. Chest X-ray imaging is a common diagnostic tool, and deep learning has shown great promise in automating the interpretation of these images.

This project develops a deep learning model that classifies chest X-rays as either "NORMAL" or "PNEUMONIA".

**Features**
**Transfer Learning with ResNet18:** Utilizes a pre-trained ResNet18 model on ImageNet as a starting point for faster convergence and improved performance on a relatively smaller dataset.
**Data Augmentation:** Includes image transformations such as resizing, random horizontal flips, and random rotations to enhance model generalization.
**Class Imbalance Handling:** Uses a weighted Cross-Entropy Loss function to account for more pneumonia cases than normal cases in the dataset.
**Robust Evaluation:** Evaluates performance using a confusion matrix, classification report (precision, recall, F1-score), and accuracy on a separate test set.
**Interactive Gradio Demo:** Provides a simple web interface for uploading chest X-ray images and receiving instant predictions from the trained model.
**Model Checkpointing:** Saves the best performing model based on validation accuracy during training.
**Dataset**
The dataset used for this project is the Chest X-Ray Images (Pneumonia) dataset from Kaggle.
Content: The dataset is organized into train/ and test/ sets, with subfolders for NORMAL and PNEUMONIA cases.

Note: No separate validation set is provided, so the training data is split to create one.

**Model Architecture**
The model is based on ResNet18.
The pre-trained convolutional layers of ResNet18 are frozen to leverage the learned features.
Custom fully connected layers are added for binary classification:
A linear layer to reduce feature dimensions.
A ReLU activation function.
A Dropout layer for regularization.
A final linear layer with 2 output units (NORMAL, PNEUMONIA).

**Training**
Optimizer: Adam
Loss Function: Cross-Entropy Loss (weighted to handle class imbalance)
Scheduler: ReduceLROnPlateau (adjusts learning rate based on validation accuracy)

**Evaluation**
The trained model is tested on an independent test set. The following metrics are used to evaluate performance:
Accuracy: Overall percentage of correct predictions.
Confusion Matrix: Counts of true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN) predictions.
Classification Report: Precision, recall, and F1-score for both classes.

**Gradio Demo**
An interactive web app is included for quick testing of the model. Simply upload a chest X-ray image and receive predictions directly from the trained model.
