import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def stability_training(X, y, fractions):
    """
    Train XGBoost models on progressively larger chunks of data.

    Parameters:
    ------------
    X : array-like (n_samples, n_features)
        Feature matrix.
    y : array-like (n_samples,)
        Target labels.
    fractions : list of floats
        Fractions of the dataset to use cumulatively (must sum to 1).

    Returns:
    ---------
    results : list of dicts
        Each dict contains { 'fraction': float, 'accuracy': float,
                             'feature_importances': np.array,
                             'distance_from_prev': float or None }
    """
    results = []
    n_samples = X.shape[0]

    # always use a test set for evaluation
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    n_train = X_train_full.shape[0]

    cum_frac = 0
    prev_importances = None

    for frac in fractions:
        cum_frac += frac
        subset_size = int(n_train * cum_frac)

        # slice subset
        X_sub, y_sub = X_train_full[:subset_size], y_train_full[:subset_size]

        # train model
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        model.fit(X_sub, y_sub)

        # evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # get feature importances
        importances = model.feature_importances_

        # compute euclidean distance to previous
        if prev_importances is None:
            dist = None
        else:
            dist = np.linalg.norm(importances - prev_importances)

        results.append({
            'fraction': cum_frac,
            'accuracy': acc,
            'feature_importances': importances,
            'distance_from_prev': dist
        })

        prev_importances = importances

    return results


def stability_plots(results):
    """
    Visualize model stability by plotting accuracy and distances.

    Parameters:
    ------------
    results : list of dicts
        Output of incremental_training
    """
    fractions = [r['fraction'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    distances = [r['distance_from_prev'] for r in results]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = 'tab:blue'
    ax1.set_xlabel('Fraction of Training Data Used')
    ax1.set_ylabel('Accuracy', color=color)
    ax1.plot(fractions, accuracies, marker='o', color=color, label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 1)

    ax2 = ax1.twinx()  # instantiate a second y-axis
    color = 'tab:red'
    ax2.set_ylabel('Euclidean Distance', color=color)
    ax2.plot(fractions, distances, marker='s', linestyle='--', color=color, label='Distance')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Model Stability Across Incremental Training')
    fig.tight_layout()
    plt.show()


