import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('house.csv')

# Display the first few rows of the dataset
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Select independent and dependent variables
# Assuming the target variable is 'Price' and using first 5 columns as features
X = data.iloc[:, :-1]
y = data['Price']

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Perform predictions
y_pred = model.predict(X_test)

# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Calculate accuracy score (R^2 Score)
r2 = r2_score(y_test, y_pred)
print(f"R^2 Score: {r2}")

# Print coefficients and intercept
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Plot the linear model
plt.figure(figsize=(15, 10))
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.scatter(X_test.iloc[:, i], y_test, color='blue', label='Actual')
    plt.scatter(X_test.iloc[:, i], y_pred, color='red', label='Predicted')
    plt.title(f"Feature {X.columns[i]} vs Price")
    plt.xlabel(X.columns[i])
    plt.ylabel('Price')
    plt.legend()
plt.tight_layout()
plt.show()
