import matplotlib.pyplot as plt
import numpy as np
import shap
from lime.lime_tabular import LimeTabularExplainer


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
    # select a subset of rows for clarity
    df = df.copy()
    if len(df) > 4:
        df = df.sample(n=4, random_state=42)
    for _, row in df.iterrows():
        X_row = row.to_frame().T
        max = df[feature].max()
        min = df[feature].min()
        feature_values = np.linspace(min, max, num=4)
        feature_values, preds = ICE(model, feature, X_row, feature_values)
        plt.plot(feature_values, preds, alpha=0.3)
    plt.title(f"Individual Conditional Expectation (ICE) for {feature}")
    plt.xlabel(feature)
    plt.ylabel("Predicted Probability")
    plt.grid()
    plt.show()


def lime_interpreter(X_train):
    lime_explainer = LimeTabularExplainer(
        X_train.values,
        feature_names=X_train.columns,
        discretize_continuous=True,
        mode="classification",
        categorical_features=np.where(X_train.nunique() <= 2)[0].tolist(),
        discretizer="decile",
    )
    return lime_explainer


def lime_interpretation(lime_explainer, model, instance, show=True, filename=None):
    lime_explanation = lime_explainer.explain_instance(
        instance.values, model.predict_proba, num_features=10
    )
    if show:
        lime_explanation.as_pyplot_figure()
    if filename:
        lime_explanation.save_to_file(filename)
    return lime_explanation


def shap_interpretation(model, X_train, instance):
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(instance)
    shap.initjs()
    return shap_values


def shap_plot(shap_values, plot_type="bar"):
    if plot_type == "bar":
        shap.plots.bar(shap_values)
    elif plot_type == "beeswarm":
        shap.plots.beeswarm(shap_values)
    shap.plots.initjs()
