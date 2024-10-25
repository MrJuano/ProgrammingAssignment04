# Import the necessary libraries
# Your code here
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load nbaallelo_log.csv into a dataframe
NBA = pd.read_csv('nbaallelo_log.csv') # Your code here

# Create binary feature for game_result with 0 for L and 1 for W
NBA['win'] = NBA['game_result'].apply(lambda x: 1 if x == 'W' else 0) # Your code here

# Store relevant columns as variables
X = NBA[['elo_i']]
y = NBA[['win']].values.ravel()

# Initialize and fit the logistic model using the LogisticRegression() function
# Your code here
model = LogisticRegression()
model.fit(X, y)

# Print the weights for the fitted model
print('w1:', model.coef_) # Your code here

# Print the intercept of the fitted model
print('w0:', model.intercept_) # Your code here

# Find the proportion of instances correctly classified
y_pred = model.predict(X)
score = accuracy_score(y, y_pred) # Your code here
print(round(score, 3))
