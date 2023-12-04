#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error





# Load the dataset
loan_old = pd.read_csv("loan_old.csv")
print(loan_old)
# Display basic information about the dataset
print(loan_old.info())

# i) Check for missing values
print("Missing Values:\n", loan_old.isnull().sum())

# ii) Check the type of each feature
print("Data Types:\n", loan_old.dtypes)

# iii) Check if numerical features have the same scale
print("Statistical Summary:\n", loan_old.describe())

# iv) Visualize a pairplot between numerical columns
sns.pairplot(loan_old.select_dtypes(include=['float64']))
plt.show()



# c) Preprocess the data

# i) Remove records containing missing values
loan_old_cleaned = loan_old.dropna()

# ii) Separate features and targets
# ii) Separate features and targets
X = loan_old_cleaned.drop(columns=['Loan_ID','Max_Loan_Amount', 'Loan_Status'])

y_amount = loan_old_cleaned['Max_Loan_Amount']
y_status = loan_old_cleaned['Loan_Status']

# iii) Shuffle and split into training and testing sets
X_train, X_test, y_amount_train, y_amount_test, y_status_train, y_status_test = train_test_split(
    X, y_amount, y_status, test_size=0.2, random_state=90
)

# iv) Categorical feature encoding
encoder = LabelEncoder()
categorical_cols = X.select_dtypes(include=['object']).columns
for col in categorical_cols:
    X_train[col] = encoder.fit_transform(X_train[col])
    X_test[col] = encoder.transform(X_test[col])

# v) Categorical targets encoding
le_status = LabelEncoder()
y_status_train_encoded = le_status.fit_transform(y_status_train)
y_status_test_encoded = le_status.transform(y_status_test)

# vi) Numerical feature standardization
scaler = StandardScaler()
numerical_cols = X.select_dtypes(include=['float64']).columns
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])




# d) Fit a linear regression model to predict the loan amount
# -> Use sklearn's linear regression
linear_reg_model = LinearRegression()
linear_reg_model.fit(X_train, y_amount_train)





# e) Evaluate the linear regression model using sklearn's R2 score and Mean Squared Error
y_amount_pred = linear_reg_model.predict(X_test)
r2_score_result = r2_score(y_amount_test, y_amount_pred)
mse_result = mean_squared_error(y_amount_test, y_amount_pred)
print(f"R^2 Score for Linear Regression: {r2_score_result}")
print(f"Mean Squared Error for Linear Regression: {mse_result}")





# f) Fit a logistic regression model to predict the loan status
# -> Implement logistic regression from scratch using gradient descent

# Define the sigmoid function
def sigmoid(z):
    # Clip the input to the exponential function within a certain range
    z = np.clip(z, -700, 700)
    return 1 / (1 + np.exp(-z))


# Implement logistic regression from scratch using gradient descent
def logistic_regression(X, y, learning_rate=0.01, epochs=1000):
    m, n = X.shape
    X = np.c_[np.ones((m, 1)), X]  # add intercept term

    theta = np.zeros(n + 1)

    for epoch in range(epochs):
        z = np.dot(X, theta)
        h = sigmoid(z)
        gradient = np.dot(X.T, (h - y)) / m
        gradient = np.clip(gradient, -10, 10)
        theta -= learning_rate * gradient

    return theta

# Train logistic regression model
theta = logistic_regression(X_train, y_status_train_encoded)
print(theta)





# g) Write a function (from scratch) to calculate the accuracy of the model
def predict(X, theta):
    X = np.c_[np.ones((X.shape[0], 1)), X]  # add intercept term
    probabilities = sigmoid(np.dot(X, theta))
    #print(probabilities)
    predictions = (probabilities >= 0.5).astype(int)
    return predictions

# Calculate accuracy on the test set
X_test_intercept = np.c_[np.ones((X_test.shape[0], 1)), X_test]
y_status_pred = predict(X_test, theta)
accuracy = np.mean(y_status_pred == y_status_test_encoded)
#print(accuracy)
print(f"Accuracy for Logistic Regression: {accuracy*100} %")
print(y_status_pred)


#loan_new_cleaned=pd.read_csv("loan_new.csv")

#loan_new_cleaned.to_csv(output,index=false)
#x=loan_new_cleaned.iloc[:, 1:].values
#print(2)
loan_new= pd.read_csv("loan_new.csv") #h
# i) Remove records containing missing values
loan_new_cleaned = loan_new.dropna()
loan_new_cleaned = loan_new_cleaned.drop(columns=['Loan_ID'])

# iv) Categorical feature encoding
encoder = LabelEncoder()
categorical_cols = loan_new_cleaned.select_dtypes(include=['object']).columns
for col in categorical_cols:
    loan_new_cleaned[col] = encoder.fit_transform(loan_new_cleaned[col])

# v) Categorical targets encoding
le_status = LabelEncoder()
y_status_encoded = le_status.fit_transform(y_status)

# vi) Numerical feature standardization
scaler = StandardScaler()
numerical_cols = loan_new_cleaned.select_dtypes(include=['float64']).columns
loan_new_cleaned[numerical_cols] = scaler.fit_transform(loan_new_cleaned[numerical_cols])




print(loan_new_cleaned)

new_loan_amount_pred = linear_reg_model.predict(loan_new_cleaned)
new_loan_status_pred = predict(loan_new_cleaned, theta)

# Display the predictions
print("Predicted Loan Amounts:")
print(new_loan_amount_pred)

print("\nPredicted Loan Status:")
print(new_loan_status_pred)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




