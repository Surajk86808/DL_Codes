# Breast Cancer Prediction Using Neural Networks

This project implements a neural network model to predict whether breast cancer is malignant or benign using the Wisconsin Breast Cancer dataset.

## Overview

The model uses a feedforward neural network built with TensorFlow/Keras to classify breast cancer tumors based on various cell nucleus features extracted from fine needle aspiration (FNA) images.

## Dataset

- **Source**: Wisconsin Breast Cancer (Diagnostic) Dataset from scikit-learn
- **Samples**: 569 instances
- **Features**: 30 numeric features describing cell nucleus characteristics
- **Classes**: 
  - 0: Malignant (212 instances)
  - 1: Benign (357 instances)

### Features Include:
- Mean, standard error, and "worst" values for:
  - Radius, texture, perimeter, area
  - Smoothness, compactness, concavity
  - Concave points, symmetry, fractal dimension

## Model Architecture

```
Sequential Model:
├── Flatten Layer (input_shape: 30)
├── Dense Layer (20 neurons, ReLU activation)
└── Dense Layer (2 neurons, Sigmoid activation)
```

## Implementation Details

### Data Preprocessing
- Feature standardization using `StandardScaler`
- Train-test split: 80%-20%
- Validation split: 10% of training data

### Training Configuration
- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy
- **Metrics**: Accuracy
- **Epochs**: 10
- **Batch Size**: Default

### Performance
- **Test Accuracy**: 97.37%
- **Training Progress**: Steady improvement from ~38% to ~97% accuracy
- **Validation Accuracy**: Reaches 95.65% by final epoch

## Results

The model demonstrates excellent performance in distinguishing between malignant and benign breast cancer cases:

- High accuracy on both training and test sets
- Good generalization with minimal overfitting
- Stable training progression over 10 epochs

## Usage

### Running the Model

1. **Load and preprocess data**:
```python
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

breast_cancer_dataset = load_breast_cancer()
# Data preprocessing steps...
```

2. **Train the model**:
```python
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(30,)),
    keras.layers.Dense(20, activation='relu'),
    keras.layers.Dense(2, activation='sigmoid')
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train_std, Y_train, validation_split=0.1, epochs=10)
```

3. **Make predictions**:
```python
predictions = model.predict(X_test_std)
predicted_labels = [np.argmax(pred) for pred in predictions]
```

### Example Prediction

```python
# Sample input (30 features)
input_data = (17.99, 10.38, 122.8, 1001.0, 0.11840, ...)
# Standardize and predict
prediction = model.predict(standardized_input)
# Output: 0 (Malignant) or 1 (Benign)
```

## Requirements

- Python 3.x
- TensorFlow/Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Files Structure

```
├── Breast_cancer_prediction.ipynb    # Main notebook
├── README.md                         # This file
```

## Important Notes

⚠️ **Medical Disclaimer**: This model is for educational and research purposes only. It should not be used for actual medical diagnosis. Always consult qualified healthcare professionals for medical decisions.

## Future Improvements

- Hyperparameter tuning
- Cross-validation
- Feature selection analysis
- Model interpretability (SHAP values)
- Ensemble methods
- Regularization techniques

## References

- [Wisconsin Breast Cancer Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)
- Original paper: W.N. Street, W.H. Wolberg and O.L. Mangasarian. Nuclear feature extraction for breast tumor diagnosis. IS&T/SPIE 1993 International Symposium on Electronic Imaging: Science and Technology, volume 1905, pages 861-870, San Jose, CA, 1993.
