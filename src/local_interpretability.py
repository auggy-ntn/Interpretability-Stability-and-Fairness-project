import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.inspection import PartialDependenceDisplay
from lime.lime_tabular import LimeTabularExplainer


def plot_ice(estimator, X, feature, *, centered=False,
             target=None, grid_resolution=30, ax=None, title=None):
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
        centered=centered
    )
    if title:
        disp.axes_[0, 0].set_title(title)
    plt.tight_layout()
    return disp


def plot_ice_subsampled(estimator, X, feature, *, subsample=200,
                        random_state=0, centered=False, target=None,
                        grid_resolution=50, alpha=0.15, linewidth=0.8, ax=None,
                        title=None):
    """
    1D ICE with subsampling for readability.
    `feature` can be a column name or index.
    """
    disp = PartialDependenceDisplay.from_estimator(
        estimator,
        X,
        features=[feature],
        kind="individual",   # ICE
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

    if title:
        ax0.set_title(title)
    plt.tight_layout()
    return ax0


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
