import pandas as pd
from sklearn.linear_model import LinearRegression
from data_setup import load_data, train_test_split_concrete

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

# Multivariate model
multi_model = LinearRegression()
multi_model.fit(X_train, y_train)
multi_coefficients = multi_model.coef_

# Univariate models
uni_coefficients = []
for i in range(8):
    X_feature = X_train[:, i].reshape(-1, 1)
    model = LinearRegression()
    model.fit(X_feature, y_train)
    uni_coefficients.append(model.coef_[0])


# Put coefficients into a table
results = pd.DataFrame({
    "Feature": feature_names,
    "Univariate Coefficient": uni_coefficients,
    "Multivariate Coefficient": multi_coefficients
})

results = results.round(6)
print(results)
results.to_csv("q6_coefficient_comparison.csv", index=False)