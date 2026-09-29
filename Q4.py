import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("Concrete_Data.csv")

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values


# rows 501-630 are test data
X_test = X[500:630]
y_test = y[500:630]

X_train = np.concatenate((X[:500], X[630:]))
y_train = np.concatenate((y[:500], y[630:]))


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def r_squared(y_true, y_pred):
    return 1 - mse(y_true, y_pred) / np.var(y_true)


# untransformed model
model_raw = LinearRegression()
model_raw.fit(X_train, y_train)

raw_train_pred = model_raw.predict(X_train)
raw_test_pred = model_raw.predict(X_test)


# standardized model
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_scaled = LinearRegression()
model_scaled.fit(X_train_scaled, y_train)

scaled_train_pred = model_scaled.predict(X_train_scaled)
scaled_test_pred = model_scaled.predict(X_test_scaled)


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