import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np



def PDF_values(df, model, feature):
    """Compute the Partial Dependence Function (PDF) values for given features.
    Args:
        df (pd.DataFrame): The input DataFrame containing features.
        model: A trained model with a predict_proba method.
        feature (str or tuple): The feature(s) for which to compute the PDF.
    Returns:
        tuple: containing the unique feature values and their corresponding average predicted probabilities.
    """
    # ATTENTION: models MUST have a predict_proba method
    # Dual feature case
    if len(feature) == 2:
        feat1, feat2 = feature
        feature_values_1 = df.unique_values(feat1)
        feature_values_1 = np.sort(feature_values_1)
        PDF = ([], [])
        for val1 in feature_values_1:
            df1 = df.copy()
            df1[feat1] = val1
            pdf = PDF_values(df1, model, feat2)
            for val2, prob in zip(pdf[0], pdf[1]):
                PDF[0].append((val1, val2))
                PDF[1].append(prob)
        return PDF
    
    # Single feature case
    feature_values = df.unique_values(feature)
    feature_values = np.sort(feature_values)
    PDF = ([], [])
    for val in feature_values:
        PDF[0].append(val)
        X = df.copy()
        X[feature] = val
        y = model.predict_proba(X)
        PDF[1].append(y[:, 1].mean())
    return PDF


def PDP_plot(PDF, feature):
    """Plot the Partial Dependence Plot (PDP) for given features."""
    if len(feature) == 2:
        PDP_plot_2D(PDF, feature)
    else:
        PDP_plot_1D(PDF, feature)


def PDP_plot_2D(PDF, feature):
    """Plot the 2D Partial Dependence Plot (PDP) for two features."""
    X, Y = zip(*PDF[0])
    Z = PDF[1]
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, Z, c='r', marker='o')
    ax.set_xlabel(feature[0])
    ax.set_ylabel(feature[1])
    ax.set_zlabel('Average Predicted Probability')
    plt.title(f'Partial Dependence Plot for {feature[0]} and {feature[1]}')
    plt.show()


def PDP_plot_1D(PDF, feature):
    """Plot the 1D Partial Dependence Plot (PDP) for a single feature."""
    plt.figure(figsize=(8, 6))
    plt.plot(PDF[0], PDF[1], marker='o')
    plt.title(f'Partial Dependence Plot for {feature}')
    plt.xlabel(feature)
    plt.ylabel('Average Predicted Probability')
    plt.grid()
    plt.show()