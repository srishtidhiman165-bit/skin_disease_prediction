
# Skin Disease Prediction Using Deep Learning

## 1. Project Overview

Skin Disease Prediction is a deep learning-based image classification project
developed using the HAM10000 skin lesion dataset.

The objective of this project is to classify skin lesion images into seven
different disease categories using the EfficientNetB0 convolutional neural
network architecture.

## 2. Dataset

The project uses the HAM10000 dataset.

The dataset contains 10,015 skin lesion images belonging to seven classes.

### Classes

1. Actinic keratoses
2. Basal cell carcinoma
3. Benign keratosis
4. Dermatofibroma
5. Melanocytic nevi
6. Melanoma
7. Vascular lesions

## 3. Technology Used

- Python
- TensorFlow
- Keras
- EfficientNetB0
- NumPy
- Pandas
- Scikit-learn
- Pillow
- Matplotlib
- VS Code

## 4. Methodology

The project follows these steps:

1. Dataset collection
2. Dataset organization
3. Train-validation-test splitting
4. Data augmentation
5. Model development
6. Model training
7. Model evaluation
8. Single-image prediction

## 5. Model

EfficientNetB0 pretrained on ImageNet was used as the base model.

The base model was initially frozen and additional classification layers
were added for the seven skin disease classes.

Input image size:

224 x 224 pixels

Number of classes:

7

## 6. Results

The trained model achieved:

Test Accuracy: 79.32%

A separate single-image prediction was also performed.

Actual Disease:
Melanocytic nevi

Predicted Disease:
Melanocytic nevi

Prediction Confidence:
96.05%

The prediction was correct for the tested HAM10000 image.

## 7. Evaluation

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## 8. Conclusion

The project demonstrates how deep learning can be used for automated
classification of skin lesion images.

The EfficientNetB0-based model achieved 79.32% accuracy on the test dataset
and successfully classified an example image as Melanocytic nevi.

## 9. Disclaimer

This project is intended for educational and research purposes only.
The predictions should not be considered medical advice or a clinical diagnosis.
Professional medical evaluation is required for actual diagnosis.