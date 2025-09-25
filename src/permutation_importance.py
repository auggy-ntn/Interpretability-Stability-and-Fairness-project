import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import permutation_importance


def plot_permutation_importance(
    model, X, y, metric="accuracy", title="Permutation Importance", S=10
):
    """
    Plot the permutation importance scores.

    Parameters:
    perm_importance (pd.Series): A Series containing the importance scores for each feature.
    title (str): The title of the plot.
    """

    perm_importance = permutation_importance(model, X, y, scoring=metric, n_repeats=S)
    relative_perm_importance = perm_importance.importances_mean / np.sum(
        perm_importance.importances_mean
    )

    sorted_idx = np.argsort(relative_perm_importance)

    plt.figure(figsize=(10, 6))
    plt.barh(X.columns[sorted_idx], relative_perm_importance[sorted_idx])
    plt.title(title)
    plt.ylabel("Features")
    plt.xlabel("Relative Importance Score")
    plt.grid()
    plt.show()
