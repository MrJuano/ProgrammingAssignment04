import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier

mpg = pd.read_csv('mpg.csv')

# Create a dataframe X containing cylinders, weight, and mpg
# Your code here
X = mpg[['cylinders', 'weight', 'mpg']]

# Create a dataframe y containing origin
# Your code here
y = mpg['origin']

# Get user-input learning rate
lr = float(input())

# Initialize and fit an adaptive boosting classifier with the user-input learning rate and a 
# random state of 123
adaBoostModel = AdaBoostClassifier(learning_rate=lr, random_state=123) # Your code here
# Your code here
adaBoostModel.fit(X, y)

# Initialize and fit a gradient boosting classifier with the user-input learning rate and a 
# random state of 123
gradientBoostModel = GradientBoostingClassifier(learning_rate=lr, random_state=123) # Your code here
# Your code here
gradientBoostModel.fit(X, y)

# Calculate the prediction accuracy for the adaptive boosting classifier
adaBoostScore = adaBoostModel.score(X, y) # Your code here
print(round(adaBoostScore, 4))

# Calculate the prediction accuracy for the gradient boosting classifier
gradientBoostScore = gradientBoostModel.score(X, y) # Your code here
print(round(gradientBoostScore, 4))
