# Loan Prediction Project

## Overview
This project aims to predict loan amounts and loan approval status based on various factors using machine learning techniques. It includes data preprocessing, model training, and evaluation.

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Models](#models)
6. [Results](#results)
7. [Contributing](#contributing)
8. [License](#license)

## Introduction
In this project, we utilize machine learning algorithms to predict loan amounts and loan approval status based on customer information such as income, education, credit history, etc. The project involves data preprocessing, model training, and evaluation.

## Dataset
The dataset used in this project contains information about loan applicants, including features like income, education, credit history, etc. It is provided in a CSV format and can be found in the repository as `loan_old.csv`.

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/your_username/loan-prediction.git
   ```
2. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
1. Navigate to the project directory:
   ```sh
   cd loan-prediction
   ```
2. Run the Jupyter notebook `Loan_Prediction.ipynb` to train and evaluate the models.

## Models
Two types of models are implemented in this project:
- Linear Regression: Used to predict the loan amount.
- Logistic Regression: Used to predict the loan approval status.

## Results
The performance of the models is evaluated using the following metrics:
- R^2 Score for Linear Regression
- Mean Squared Error for Linear Regression
- Accuracy for Logistic Regression
