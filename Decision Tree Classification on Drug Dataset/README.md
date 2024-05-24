# Decision Tree Classification on Drug Dataset

## Overview
This project demonstrates the use of Decision Tree classification to predict the type of drug prescribed based on various patient attributes. The project includes data preprocessing, model training, evaluation across different random states and training set sizes, and visualization of results.

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Functions and Key Operations](#functions-and-key-operations)
6. [Results](#results)


## Introduction
The objective of this project is to build a Decision Tree classifier to predict the type of drug prescribed to a patient based on several attributes such as age, sex, blood pressure, and cholesterol levels. The dataset used is `drug.csv`, and the code involves data preprocessing, encoding, model training, evaluation, and result visualization.

## Dataset
The dataset `drug.csv` contains the following columns:
- Age
- Sex
- BP (Blood Pressure)
- Cholesterol
- Na_to_K (Sodium to Potassium ratio)
- Drug (Target variable)

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/your_username/decision-tree-drug-prediction.git
   ```
2. Navigate to the project directory:
   ```sh
   cd decision-tree-drug-prediction
   ```
3. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
1. Ensure the dataset `drug.csv` is in the project directory.
2. Run the Python script:
   ```sh
   python decision_tree_drug.py
   ```
3. The script will output the evaluation results of the Decision Tree classifier across different experiments and training set sizes.

## Functions and Key Operations
- **Data Preprocessing**: Handles missing values, encodes categorical features, and standardizes numerical features.
- **train_test_split**: Splits the data into training and testing sets.
- **DecisionTreeClassifier**: Trains the Decision Tree model.
- **accuracy_score**: Evaluates the accuracy of the model.

### Experiment 1: Varying Random States
- Trains and evaluates the model using different random states.
- Prints the size of the decision tree and accuracy for each experiment.
- Identifies and prints the best performing model.

### Experiment 2: Varying Training Set Sizes
- Trains and evaluates the model using different training set sizes.
- Prints the mean, maximum, and minimum accuracy and tree size for each training set size.
- Visualizes the results using matplotlib.

## Results
The script evaluates the Decision Tree classifier across different random states and training set sizes, providing insights into the model's performance.

### Example Output
```
Experiment no. 1:
Size of Decision Tree: 31
Accuracy equals: 0.94
---------------------------------------------------------------------
The Best Performing Model is from Experiment number: 1
It's Size: 31
It's Accuracy: 0.94
```

### Plots
- **Accuracy vs Training Set Size**
- **Tree Size vs Training Set Size**

The plots provide a visual representation of how the accuracy and tree size vary with different training set sizes.

