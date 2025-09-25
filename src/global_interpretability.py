import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import PartialDependenceDisplay


def plot_pdp_1d(estimator, X, feature, *, target=None,
                grid_resolution=30, ax=None, title=None, xticklabels=None):
    """
    1D Partial Dependence Plot (PDP) for a single feature.

    Params
    - estimator: fitted model (sklearn API, e.g. XGBClassifier)
    - X: pandas DataFrame or array (same preprocessing the model expects)
    - feature: str | int (column name or index)
    - target: class index for multiclass (None for binary)
    - grid_resolution: number of evaluation points
    - ax: optional matplotlib Axes
    - title: optional string for plot title
    - xticklabels: optional list of labels for the x-axis ticks (e.g. ['A','B','C','D','E','F'])
    """
    disp = PartialDependenceDisplay.from_estimator(
        estimator,
        X,
        features=[feature],
        kind="average",
        target=target,
        grid_resolution=grid_resolution,
        ax=ax
    )
    if title:
        disp.axes_[0, 0].set_title(title)
    plt.tight_layout()
    return disp


def plot_pdp_2d(estimator, X, features_pair, *,
                target=None, grid_resolution=10, ax=None, title=None):
    """
    2D PDP for a pair of features.

    Params
    - features_pair: tuple[str|int, str|int], e.g. ('age','income') or (0,1)
    """
    disp = PartialDependenceDisplay.from_estimator(
        estimator,
        X,
        features=[tuple(features_pair)],
        kind="average",               # sklearn only supports average for 2D
        target=target,
        grid_resolution=grid_resolution,
        ax=ax
    )
    if title:
        disp.axes_[0, 0].set_title(title)
    plt.tight_layout()
    return disp
