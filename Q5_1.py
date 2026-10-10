import matplotlib.pyplot as plt
from data_setup import (load_data, train_test_split_concrete, standardize_data, log_transform)
from gradient_descent import (gradient_descent, predict, r_squared)

# Load and split data
X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split_concrete(X, y)

# Untransformed model
raw_weights, raw_loss = gradient_descent(X_train, y_train, learning_rate=3e-7, n_iterations=1000)

# Standardized model
X_train_scaled, X_test_scaled = standardize_data(X_train, X_test)
scaled_weights, scaled_loss = gradient_descent(X_train_scaled, y_train, learning_rate=0.01, n_iterations=1000)

# Log transformed model
X_train_log, X_test_log = log_transform(X_train, X_test)
log_weights, log_loss = gradient_descent(X_train_log, y_train, learning_rate=0.005, n_iterations=1000)

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