# Data Preprocessing Tools
# Importing the libraries
# matplotlib hasn't been used
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

### The Data Set is 768 rows and 9 columns (outcome included)
# Importing the dataset
dataset = pd.read_csv('Blood Pressure.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

### Couldn't find any missing data in the data set I got but I did the fixing anyway
# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, :-1])
X[:, :-1] = imputer.transform(X[:, :-1])

### No Need For Dummy Variables
### No need to do Enconding because there is no data with n known value
### All the data are in ranges
# Encoding categorical data
# Encoding the Independent Variable
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder
# ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
# X = np.array(ct.fit_transform(X))

### No need to do Label Encoding because there's no (yes, no) data like
### The outcome of the data is already (0,1) Based so there's no need to Encode it
# Encoding the Dependent Variable
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# y = le.fit_transform(y)


### Splitting Values are (30% Testing and 70% Training)
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


### all incomes need to be Scaled because it's range based values
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, :] = sc.fit_transform(X_train[:, :])
X_test[:, :] = sc.transform(X_test[:, :])


### Exporting the Results to a new CSV file
data = pd.DataFrame(X_train, None, ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                                    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"])
data.to_csv(r"C:\Users\MaginaPro\Desktop\pythonProject\PreProcessedData.csv")