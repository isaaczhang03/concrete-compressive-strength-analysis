import pandas as pd
from sklearn.linear_model import LinearRegression
from data_setup import (load_data, train_test_split_concrete, standardize_data, log_transform)
from gradient_descent import (mse, r_squared)

# Load and split data
X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split_concrete(X, y)

# Untransformed model
raw_model = LinearRegression()
raw_model.fit(X_train, y_train)

raw_train_pred = raw_model.predict(X_train)
raw_test_pred = raw_model.predict(X_test)

# Standardized model
X_train_scaled, X_test_scaled = standardize_data(X_train, X_test)
scaled_model = LinearRegression()
scaled_model.fit(X_train_scaled, y_train)
scaled_train_pred = scaled_model.predict(X_train_scaled)
scaled_test_pred = scaled_model.predict(X_test_scaled)

X_train_log, X_test_log = log_transform(X_train, X_test)

log_model = LinearRegression()
log_model.fit(X_train_log, y_train)
log_train_pred = log_model.predict(X_train_log)
log_test_pred = log_model.predict(X_test_log)

results = pd.DataFrame({
    "Model": [
        "Untransformed",
        "Standardized",
        "Log-Transformed"
    ],
    "Training MSE": [
        mse(y_train, raw_train_pred),
        mse(y_train, scaled_train_pred),
        mse(y_train, log_train_pred)
    ],
    "Testing MSE": [
        mse(y_test, raw_test_pred),
        mse(y_test, scaled_test_pred),
        mse(y_test, log_test_pred)
    ],
    "Training R2": [
        r_squared(y_train, raw_train_pred),
        r_squared(y_train, scaled_train_pred),
        r_squared(y_train, log_train_pred)
    ],
    "Testing R2": [
        r_squared(y_test, raw_test_pred),
        r_squared(y_test, scaled_test_pred),
        r_squared(y_test, log_test_pred)
    ]
})

results = results.round(4)
results.to_csv("q5_3_summary_table.csv", index=False)
