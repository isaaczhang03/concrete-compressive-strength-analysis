import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data():
    df = pd.read_csv("Concrete_Data.csv")
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

def train_test_split_concrete(X, y):
    # Rows 501-630 are the test set
    X_test = X[500:630]
    y_test = y[500:630]
    X_train = np.concatenate((X[:500], X[630:]))
    y_train = np.concatenate((y[:500], y[630:]))
    return X_train, X_test, y_train, y_test

def standardize_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def log_transform(X_train, X_test):
    X_train_log = np.log(X_train + 1)
    X_test_log = np.log(X_test + 1)
    return X_train_log, X_test_log