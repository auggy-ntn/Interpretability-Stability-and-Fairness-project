import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier


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


class LogisticRegressionSurrogate:
    def __init__(self):
        self.model = LogisticRegression()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def get_coefficients(self):
        return self.model.coef_[0]

    def plot_interpretation(self, features, max_features=10):
        coef = self.get_coefficients()
        indices = np.argsort(np.abs(coef))
        indices = indices[-max_features:]

        plt.figure(figsize=(10, 6))
        plt.barh(range(max_features), coef[indices], align="center")
        plt.yticks(range(max_features), [features[i] for i in indices])
        plt.xlabel("Coefficient Value")
        plt.title("Feature Importance from Logistic Regression Surrogate Model")
        plt.show()

        indices_pos = np.argsort(coef)
        indices_pos = indices_pos[-max_features:]

        plt.figure(figsize=(10, 6))
        plt.barh(range(max_features), coef[indices_pos], align="center")
        plt.yticks(range(max_features), [features[i] for i in indices_pos])
        plt.xlabel("Coefficient Value")
        plt.title(
            "Positive Feature Importance from Logistic Regression Surrogate Model"
        )
        plt.show()

        indices_neg = np.argsort(-coef)
        indices_neg = indices_neg[-max_features:]

        plt.figure(figsize=(10, 6))
        plt.barh(range(max_features), coef[indices_neg], align="center")
        plt.yticks(range(max_features), [features[i] for i in indices_neg])
        plt.xlabel("Coefficient Value")
        plt.title(
            "Negative Feature Importance from Logistic Regression Surrogate Model"
        )
        plt.show()


class DecisionTreeSurrogate:
    def __init__(self, max_depth=3):
        self.model = DecisionTreeClassifier(max_depth=max_depth)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def get_feature_importances(self):
        return self.model.feature_importances_

    def plot_interpretation(self, features, max_features=10):
        importances = self.get_feature_importances()
        indices = np.argsort(np.abs(importances))
        indices = indices[-max_features:]

        plt.figure(figsize=(10, 6))
        plt.barh(range(max_features), importances[indices], align="center")
        plt.yticks(range(max_features), [features[i] for i in indices])
        plt.xlabel("Coefficient Value")
        plt.title("Feature Importance from Decision Tree Surrogate Model")
        plt.show()


class RandomForestSurrogate:
    def __init__(self, n_estimators=100, max_depth=3):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def get_feature_importances(self):
        return self.model.feature_importances_

    def plot_interpretation(self, features, max_features=10):
        importances = self.get_feature_importances()
        indices = np.argsort(np.abs(importances))
        indices = indices[-max_features:]

        plt.figure(figsize=(10, 6))
        plt.barh(range(max_features), importances[indices], align="center")
        plt.yticks(range(max_features), [features[i] for i in indices])
        plt.xlabel("Coefficient Value")
        plt.title("Feature Importance from Random Forest Surrogate Model")
        plt.show()
