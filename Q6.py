from data_setup import (load_data, train_test_split_concrete)
from gradient_descent import (gradient_descent, predict, r_squared)

# Load and split data
X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split_concrete(X, y)


feature_names = [
    "Cement",
    "Blast Furnace Slag",
    "Fly Ash",
    "Water",
    "Superplasticizer",
    "Coarse Aggregate",
    "Fine Aggregate",
    "Age"
]


# Learning rates chosen to keep each untransformed univariate model stable
learning_rates = [
    1e-5,   # Cement
    1e-5,   # Blast Furnace Slag
    1e-4,   # Fly Ash
    1e-5,   # Water
    1e-4,   # Superplasticizer
    1e-6,   # Coarse Aggregate
    1e-6,   # Fine Aggregate
    1e-4    # Age
]


for i in range(8):
    weights, loss = gradient_descent(X_train[:, i], y_train, learning_rate=learning_rates[i], n_iterations=50000)
    train_pred = predict(X_train[:, i], weights)
    test_pred = predict(X_test[:, i], weights)
    train_r2 = r_squared(y_train, train_pred)
    test_r2 = r_squared(y_test, test_pred)

    print(feature_names[i])
    print("Learning rate:", learning_rates[i])
    print("Training R2:", train_r2)
    print("Testing R2:", test_r2)
    print()