# Import the necessary modules
# Your code here
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load the dataset
skySurvey = pd.read_csv('SDSS.csv') # Your code here

# Create a new feature from u - g
skySurvey['u_g'] = skySurvey['u'] - skySurvey['g']

# Create dataframe X with features redshift and u_g
X = skySurvey[['redshift', 'u_g']] # Your code here

# Create dataframe y with feature class
y = skySurvey['class'] # Your code here

# Initialize a Gaussian naive Bayes model
skySurveyNBModel = GaussianNB()# Your code here

# Fit the model
# Your code here
skySurveyNBModel.fit(X, np.ravel(y))

# Calculate the proportion of instances correctly classified
y_pred = skySurveyNBModel.predict(X)
score = accuracy_score(y, y_pred)# Your code here

# Print accuracy score
print('Accuracy score is ', end="")
print('%.3f' % score)
