import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LinearRegression


# A simple linear surrogate model for interpretability
class LinearSurrogate:
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def get_coefficients(self):
        return self.model.coef_

    def plot_interpretation(self, features, max_features=10):
        coef = self.get_coefficients()
        indices = np.argsort(coef)
        indices = indices[-max_features:]

        plt.figure(figsize=(10, 6))
        plt.barh(range(max_features), coef[indices], align="center")
        plt.yticks(range(max_features), [features[i] for i in indices])
        plt.xlabel("Coefficient Value")
        plt.title("Feature Importance from Linear Surrogate Model")
        plt.show()
