# Import needed packages for classification
# Your code here
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Import packages for evaluation
# Your code her

# Load the dataset
skySurvey = pd.read_csv('SDSS.csv')

# Create a new feature from u - g
skySurvey['u_g'] = skySurvey['u'] - skySurvey['g']

# Create dataframe X with features redshift and u_g
X = skySurvey[['redshift', 'u_g']] # Your code here

# Create dataframe y with feature class
y = skySurvey['class'] # Your code here

np.random.seed(42)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Initialize model with k=3
skySurveyKnn = KNeighborsClassifier(n_neighbors=3) # Your code here

# Fit model using X_train and y_train
skySurveyKnn.fit(X_train, y_train) # Your code here

# Find the predicted classes for X_test
y_pred = skySurveyKnn.predict(X_test) # Your code here

# Calculate accuracy score
score = accuracy_score(y_test, y_pred) # Your code here

# Print accuracy score
print('Accuracy score is ', end="")
print('%.3f' % score)
