from sklearn.linear_model import LinearRegression
from data_setup import (load_data, train_test_split_concrete, standardize_data)
from gradient_descent import (mse, r_squared)

# Load and split data
X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split_concrete(X, y)

raw_model = LinearRegression()
raw_model.fit(X_train, y_train)
raw_train_pred = raw_model.predict(X_train)
raw_test_pred = raw_model.predict(X_test)

X_train_scaled, X_test_scaled = standardize_data(X_train, X_test)
scaled_model = LinearRegression()
scaled_model.fit(X_train_scaled, y_train)
scaled_train_pred = scaled_model.predict(X_train_scaled)
scaled_test_pred = scaled_model.predict(X_test_scaled)

print("Untransformed Model")
print("Train MSE:", mse(y_train, raw_train_pred))
print("Test MSE:", mse(y_test, raw_test_pred))
print("Train R2:", r_squared(y_train, raw_train_pred))
print("Test R2:", r_squared(y_test, raw_test_pred))

print()

print("Standardized Model")
print("Train MSE:", mse(y_train, scaled_train_pred))
print("Test MSE:", mse(y_test, scaled_test_pred))
print("Train R2:", r_squared(y_train, scaled_train_pred))
print("Test R2:", r_squared(y_test, scaled_test_pred))