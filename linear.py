import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('height_weight.tsv', sep='\t')  # Or CSV

# Check for nulls
df.dropna(inplace=True)

# Select required columns
X = df[['Height']]
y = df['Weight']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# pre-precoessing part completed

# model and evaluation
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("Coefficient:", model.coef_)
print("Intercept:", model.intercept_)


# visualization
import matplotlib.pyplot as plt

plt.scatter(X_test, y_test, color='blue', label="Actual")
plt.plot(X_test, y_pred, color='red', label="Predicted")
plt.xlabel("Height")
plt.ylabel("Weight")
plt.title("Linear Regression - Height vs Weight")
plt.legend()
plt.show()
# Save the model

# do not use until soething goes wrong 
# this is  a generic code 
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, r2_score
# import matplotlib.pyplot as plt

# # Assume df with columns: 'X' and 'y'
# X = df[['X']]
# y = df['y']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# model = LinearRegression()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)

# print("MSE:", mean_squared_error(y_test, y_pred))
# print("R2:", r2_score(y_test, y_pred))

# plt.scatter(X_test, y_test, label='Actual')
# plt.plot(X_test, y_pred, color='red', label='Predicted')
# plt.legend()
# plt.show()
