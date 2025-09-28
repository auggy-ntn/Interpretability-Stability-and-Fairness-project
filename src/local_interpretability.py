import matplotlib.pyplot as plt
import numpy as np
import shap
from IPython.display import display
from lime.lime_tabular import LimeTabularExplainer
from sklearn.inspection import PartialDependenceDisplay


def plot_ice(
    estimator,
    X,
    feature,
    *,
    centered=False,
    target=None,
    grid_resolution=30,
    ax=None,
    title=None,
):
    """
    1D ICE curve(s). Use centered=True to plot centered ICE (cICE).

    Note: sklearn plots ICE via kind='individual'; cICE is achieved by
    passing 'centered=True'.
    """
    disp = PartialDependenceDisplay.from_estimator(
        estimator,
        X,
        features=[feature],
        kind="individual",
        target=target,
        grid_resolution=grid_resolution,
        ax=ax,
        centered=centered,
    )
    if title:
        disp.axes_[0, 0].set_title(title)
    plt.tight_layout()
    return disp


def plot_ice_subsampled(
    estimator,
    X,
    feature,
    *,
    subsample=200,
    random_state=0,
    centered=False,
    target=None,
    grid_resolution=50,
    alpha=0.15,
    linewidth=0.8,
    ax=None,
    title=None,
):
    """
    1D ICE with subsampling for readability.
    `feature` can be a column name or index.
    """
    disp = PartialDependenceDisplay.from_estimator(
        estimator,
        X,
        features=[feature],
        kind="individual",  # ICE
        subsample=subsample,
        random_state=random_state,
        centered=centered,
        target=target,
        grid_resolution=grid_resolution,
        ax=ax,
        line_kw={"alpha": alpha, "linewidth": linewidth},
    )

    ax0 = disp.axes_[0, 0]
    if title:
        ax0.set_title(title)

    plt.tight_layout()
    return disp


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


def shap_plot(model, data, plot_type="bar"):
    # Compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(data)
    shap.initjs()

    # Plot SHAP values
    if plot_type == "bar":
        shap.plots.bar(shap_values)
    elif plot_type == "beeswarm":
        shap.plots.beeswarm(shap_values)
    elif plot_type == "force":
        display(shap.force_plot(explainer.expected_value, shap_values.values[0], data))
