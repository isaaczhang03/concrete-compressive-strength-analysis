import pandas as pd
import statsmodels.api as sm
from data_setup import load_data, train_test_split_concrete, standardize_data, log_transform

# load data
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

# Untransformed model
X_raw = sm.add_constant(X_train)
raw_model = sm.OLS(y_train, X_raw).fit()
raw_pvalues = raw_model.pvalues[1:]

# Standardized model
X_train_scaled, X_test_scaled = standardize_data(X_train, X_test)
X_scaled = sm.add_constant(X_train_scaled)
scaled_model = sm.OLS(y_train, X_scaled).fit()
scaled_pvalues = scaled_model.pvalues[1:]

# Log transformed model
X_train_log, X_test_log = log_transform(X_train, X_test)
X_log = sm.add_constant(X_train_log)
log_model = sm.OLS(y_train, X_log).fit()
log_pvalues = log_model.pvalues[1:]

# Make table
pvalue_table = pd.DataFrame({
    "Feature": feature_names,
    "Untransformed p-value": raw_pvalues,
    "Standardized p-value": scaled_pvalues,
    "Log-Transformed p-value": log_pvalues
})

print(pvalue_table)

# Save table
pvalue_table.to_csv(
    "q7_pvalue_comparison.csv",
    index=False
)