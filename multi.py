X = df[['feature1', 'feature2', 'feature3', 'feature4', 'feature5']]
y = df['target']

# Same train_test_split as above

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
