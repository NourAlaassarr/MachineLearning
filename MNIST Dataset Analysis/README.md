# MNIST Dataset Analysis

This project analyzes the MNIST dataset using various machine learning techniques including K-Nearest Neighbors (KNN) and Artificial Neural Networks (ANN). It performs data preprocessing, model training, hyperparameter tuning, and evaluation.

## Dataset

The MNIST dataset consists of 28x28 pixel grayscale images of handwritten digits (0-9). It is commonly used for training various image processing systems.

you can download it from this link:

https://www.kaggle.com/datasets/hojjatk/mnist-dataset

## Requirements

Ensure you have the following libraries installed:

- pandas
- scikit-learn
- matplotlib
- tensorflow

You can install these libraries using pip:

```bash
pip install pandas scikit-learn matplotlib tensorflow
```

## Usage

1. Clone this repository:

```bash
git clone https://github.com/your-username/mnist-analysis.git
cd mnist-analysis
```

2. Run the Python script:

```bash
python mnist_analysis.py
```

3. Check the console output for results and follow the instructions for any further actions.

## Description

The analysis consists of the following steps:

1. **Data Loading and Exploration:** Load the MNIST dataset and explore its basic information, such as the number of classes, features, and missing values.

2. **Data Preprocessing:** Normalize the pixel values of the images to the range [0, 1] by dividing each pixel by 255.

3. **Visualization:** Display a few images from the dataset to visualize the handwritten digits.

4. **Model Training and Evaluation:**
    - Train a KNN classifier using grid search for hyperparameter tuning.
    - Train two ANN architectures and compare their performance.
    - Perform hyperparameter tuning for both architectures and select the best models.

5. **Evaluation:** Evaluate the best models on the validation set and compare their accuracies.

6. **Model Saving and Testing:** Save the best model and evaluate its performance on the test set.

## Results

The results of the analysis, including validation accuracies, confusion matrices, and test accuracy, will be displayed in the console output.

