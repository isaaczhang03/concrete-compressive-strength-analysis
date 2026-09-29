import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from gradientDescent import (
    gradient_descent,
    predict,
    r_squared
)


df = pd.read_csv("Concrete_Data.csv")

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values


# rows 501-630 are the test set
X_test = X[500:630]
y_test = y[500:630]

X_train = np.concatenate((X[:500], X[630:]))
y_train = np.concatenate((y[:500], y[630:]))



# Untransformed model
raw_weights, raw_loss = gradient_descent(
    X_train,
    y_train,
    learning_rate=3e-7,
    n_iterations=1000
)

# Standardized model
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

scaled_weights, scaled_loss = gradient_descent(
    X_train_scaled,
    y_train,
    learning_rate=0.01,
    n_iterations=1000
)

# Log transformed model
X_train_log = np.log(X_train + 1)
X_test_log = np.log(X_test + 1)

log_weights, log_loss = gradient_descent(
    X_train_log,
    y_train,
    learning_rate=0.005,
    n_iterations=1000
)


# Check R2 for log model
log_train_pred = predict(X_train_log, log_weights)
log_test_pred = predict(X_test_log, log_weights)

print("Log Transformed Model")
print("Train R2:", r_squared(y_train, log_train_pred))
print("Test R2:", r_squared(y_test, log_test_pred))

# Updated loss graph
plt.figure(figsize=(8, 5))

plt.plot(raw_loss, label="Untransformed")
plt.plot(scaled_loss, label="Standardized")
plt.plot(log_loss, label="Log Transformed")

plt.xlabel("Iteration")
plt.ylabel("MSE Loss")
plt.title("Gradient Descent Loss Over Iterations")

plt.legend()
plt.tight_layout()

plt.savefig("q5_1_loss_over_iterations.png", dpi=300)

plt.show()