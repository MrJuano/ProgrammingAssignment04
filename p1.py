# Import needed packages for regression
# Your code here
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Silence warning from sklearn
import warnings
warnings.filterwarnings('ignore')

# Input feature values for a sample instance
carat = float(input())
table = float(input())

diamonds = pd.read_csv('diamonds.csv')

# Define input and output features
X = diamonds[['carat', 'table']]
y = diamonds['price']

# Initialize a multiple linear regression model
# Your code here
model = LinearRegression()

# Fit the multiple linear regression model to the input and output features
# Your code here
model.fit(X, y)

# Get estimated intercept weight
intercept = model.intercept_ # Your code here
print('Intercept is', round(intercept, 3))

# Get estimated weights for carat and table features
coefficients = model.coef_ # Your code here
print('Weights for carat and table features are', np.round(coefficients, 3))

# Predict the price of a diamond with the user-input carat and table values
prediction = model.predict([[carat, table]]) # Your code here
print('Predicted price is', np.round(prediction, 2))
