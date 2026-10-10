import matplotlib.pyplot as plt
from data_setup import (load_data, train_test_split_concrete,standardize_data)
from gradient_descent import (gradient_descent, predict, r_squared)

# Load and split data
X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split_concrete(X, y)

# Untransformed model
raw_weights, raw_loss = gradient_descent(X_train, y_train, learning_rate=3e-7, n_iterations=1000)
raw_train_pred = predict(X_train, raw_weights)
raw_test_pred = predict(X_test, raw_weights)

print("Untransformed model")
print("Training R^2:", r_squared(y_train, raw_train_pred))
print("Testing R^2:", r_squared(y_test, raw_test_pred))

# Standardized model
X_train_scaled, X_test_scaled = standardize_data(X_train, X_test)
scaled_weights, scaled_loss = gradient_descent(X_train_scaled, y_train, learning_rate=0.01, n_iterations=1000)
scaled_train_pred = predict(X_train_scaled, scaled_weights)
scaled_test_pred = predict(X_test_scaled, scaled_weights)

print("\nStandardized model")
print("Training R^2:", r_squared(y_train, scaled_train_pred))
print("Testing R^2:", r_squared(y_test, scaled_test_pred))

# Plot loss over iterations
plt.figure(figsize=(8, 5))
plt.plot(raw_loss, label="Untransformed")
plt.plot(scaled_loss, label="Standardized")
plt.xlabel("Iteration")
plt.ylabel("MSE Loss")
plt.title("Gradient Descent Loss Over Iterations")
plt.legend()
plt.tight_layout()
plt.savefig("q3_1_loss_over_iterations.png", dpi=300)
plt.show()