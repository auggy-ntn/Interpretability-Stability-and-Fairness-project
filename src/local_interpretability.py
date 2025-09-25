import numpy as np
import matplotlib.pyplot as plt


def ICE(model, feature, X_row, feature_values):
    preds = []
    for val in feature_values:
        X_temp = X_row.copy()
        X_temp[feature] = val
        y_pred = model.predict_proba(X_temp)[:, 1][0]
        preds.append(y_pred)
    return feature_values, preds


def ICE_plot(df, model, feature):
    """Plot the Individual Conditional Expectation (ICE) curves for a given feature."""
    for _, row in df.iterrows():
        X_row = row.to_frame().T
        max = df[feature].max()
        min = df[feature].min()
        feature_values = np.linspace(min, max, num=10)
        feature_values, preds = ICE(model, feature, X_row, feature_values)
        plt.plot(feature_values, preds, alpha=0.3)
    plt.title(f'Individual Conditional Expectation (ICE) for {feature}')
    plt.xlabel(feature)
    plt.ylabel('Predicted Probability')
    plt.grid()
    plt.show()