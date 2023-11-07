import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Load the "loan_old.csv" dataset.
data=pd.read_csv('./loan_old.csv')
print(data)
print("---------------------------------")

#check whether there are missing values
print("Missing Values")
print(data.isnull().sum())
print("---------------------------------")

#check the type of each feature (categorical or numerical)
df=pd.DataFrame(data)
type=df.dtypes
print(type)
print("---------------------------------")

#check whether numerical features have the same scale
df_select = df.select_dtypes(include=['int64','float64'])
scales = df_select.describe()
print(scales)

#visualize a pairplot between numercial columns
sns.pairplot(df_select)
plt.show()