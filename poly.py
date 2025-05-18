df = pd.read_csv('house.csv')

# Drop missing
df.dropna(inplace=True)

# Select features
X = df[['area', 'bedrooms', 'bathrooms', 'location_score', 'age']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# model eval
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)


# visualization 
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
axs[0, 0].scatter(df['area'], df['price'])
axs[0, 0].set_title("Area vs Price")

axs[0, 1].scatter(df['bedrooms'], df['price'])
axs[0, 1].set_title("Bedrooms vs Price")

axs[1, 0].scatter(df['bathrooms'], df['price'])
axs[1, 0].set_title("Bathrooms vs Price")

axs[1, 1].scatter(df['age'], df['price'])
axs[1, 1].set_title("Age vs Price")

plt.tight_layout()
plt.show()
# Save the model

# this is the generic code
# from sklearn.preprocessing import PolynomialFeatures

# poly = PolynomialFeatures(degree=3)
# X_poly = poly.fit_transform(X)

# X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=1)

# model = LinearRegression()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)

# print("MSE:", mean_squared_error(y_test, y_pred))
# print("R2:", r2_score(y_test, y_pred))
