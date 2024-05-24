# Banknote Authentication and KNN Classification

This project performs Banknote Authentication using Decision Trees and analyzes the performance of K-Nearest Neighbors (KNN) classification on the Banknote Authentication dataset.

## Dataset

The Banknote Authentication dataset consists of features extracted from genuine and forged banknotes. These features include variance, skewness, curtosis, and entropy. The task is to classify whether a banknote is genuine or forged based on these features.

## Requirements

Ensure you have the following libraries installed:

- pandas
- numpy
- scikit-learn
- matplotlib

You can install these libraries using pip:

```bash
pip install pandas numpy scikit-learn matplotlib
```

## Usage

1. Clone this repository:

```bash
git clone https://github.com/your-username/banknote-authentication.git
cd banknote-authentication
```

2. Run the Python script:

```bash
python banknote_authentication.py
```

3. Check the console output for results and follow the instructions for any further actions.

## Description

The analysis consists of the following steps:

### Banknote Authentication using Decision Trees:

- Load the Banknote Authentication dataset and split it into features (X) and labels (Y).
- Train Decision Tree classifiers on different train-test splits and evaluate their accuracies.

### KNN Classification Analysis:

- Normalize the features in the dataset to ensure fair comparison.
- Split the dataset into training and testing sets.
- Evaluate KNN classifiers with different k values and analyze their performance.

