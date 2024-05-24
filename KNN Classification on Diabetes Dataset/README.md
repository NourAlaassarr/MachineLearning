# KNN Classification on Diabetes Dataset

## Overview
This project demonstrates the use of K-Nearest Neighbors (KNN) classification algorithm to predict the outcome (diabetes status) based on a dataset containing various medical measurements. The project involves data normalization, implementation of the KNN algorithm, and evaluation of the model with different k values.

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Functions](#functions)
6. [Results](#results)


## Introduction
The objective of this project is to build a KNN classifier from scratch and use it to predict whether a person has diabetes based on several medical attributes. The dataset used is the diabetes dataset, and the features are normalized using Min-Max Scaling before applying the KNN algorithm.to break ties when using the KNN algorithm, this project implements distance-weighted voting. The principle behind this approach is that closer neighbors should have a stronger influence on the classification outcome than more distant ones. The weight assigned to a neighbor is the inverse of its distance to the test instance.

## Dataset
The dataset used in this project is `diabetes.csv`, which contains the following columns:
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome (Target variable: 0 or 1)

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/your_username/knn-diabetes-prediction.git
   ```
2. Navigate to the project directory:
   ```sh
   cd knn-diabetes-prediction
   ```
3. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
1. Ensure the dataset `diabetes.csv` is in the project directory.
2. Run the Python script:
   ```sh
   python knn_diabetes.py
   ```
3. The script will output the evaluation results of the KNN classifier with different k values.

## Functions
- **min_max_scaling(data)**: Normalizes the dataset using Min-Max Scaling.
- **euclidean_distance(x1, x2)**: Computes the Euclidean distance between two points.
- **knn_classify(train_data, train_labels, test_instance, k)**: Classifies a test instance using the KNN algorithm with k neighbors.
- **train_test_split(data, labels, split_ratio=0.7)**: Splits the dataset into training and testing sets based on the given split ratio.
- **evaluate_knn(data, labels, k_values)**: Evaluates the KNN classifier with different k values and prints the accuracy.

## Results
The script evaluates the KNN classifier with different k values (26, 27, 28) and prints the following metrics for each k:
- Number of correctly classified instances
- Total number of instances
- Accuracy

Additionally, the average accuracy across all iterations is calculated and printed.
