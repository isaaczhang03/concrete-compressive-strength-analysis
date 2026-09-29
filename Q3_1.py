import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from gradientDescent import (
    gradient_descent,
    predict,
    r_squared
)


# Load data
df = pd.read_csv("Concrete_Data.csv")

# First 8 columns are predictors, last column is compressive strength
X = df.iloc[:, :-1].to_numpy(dtype=float)
y = df.iloc[:, -1].to_numpy(dtype=float)


# --------------------------------------------------
# Train/test split
# Rows 501-630 are test data
# Python indices 500-629
# --------------------------------------------------

test_mask = np.zeros(len(df), dtype=bool)
test_mask[500:630] = True

X_train = X[~test_mask]
y_train = y[~test_mask]

X_test = X[test_mask]
y_test = y[test_mask]


# --------------------------------------------------
# 1. Untransformed model
# --------------------------------------------------

raw_weights, raw_loss = gradient_descent(
    X_train,
    y_train,
    learning_rate=3e-7,
    n_iterations=1000
)

raw_train_pred = predict(X_train, raw_weights)
raw_test_pred = predict(X_test, raw_weights)

print("Untransformed model")
print("Training R^2:", r_squared(y_train, raw_train_pred))
print("Testing R^2:", r_squared(y_test, raw_test_pred))


# --------------------------------------------------
# 2. Standardized model
# --------------------------------------------------

scaler = StandardScaler()

# Fit ONLY on training data
X_train_scaled = scaler.fit_transform(X_train)

# Apply same transformation to test data
X_test_scaled = scaler.transform(X_test)

scaled_weights, scaled_loss = gradient_descent(
    X_train_scaled,
    y_train,
    learning_rate=0.01,
    n_iterations=1000
)

scaled_train_pred = predict(X_train_scaled, scaled_weights)
scaled_test_pred = predict(X_test_scaled, scaled_weights)

print("\nStandardized model")
print("Training R^2:", r_squared(y_train, scaled_train_pred))
print("Testing R^2:", r_squared(y_test, scaled_test_pred))


# --------------------------------------------------
# Plot loss over iterations
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    raw_loss,
    label="Untransformed"
)

plt.plot(
    scaled_loss,
    label="Standardized"
)

plt.xlabel("Iteration")
plt.ylabel("MSE Loss")
plt.title("Gradient Descent Loss Over Iterations")

plt.legend()
plt.tight_layout()

plt.savefig(
    "q3_1_loss_over_iterations.png",
    dpi=300
)

plt.show()