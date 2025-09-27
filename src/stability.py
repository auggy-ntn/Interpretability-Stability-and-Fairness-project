import numpy as np
from src.black_box_optimal import train_xgboost_optimal
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd


def stability_training(X, y, fractions):
    """
    Train XGBoost models with optimal parameters on progressively larger chunks of data.
    
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
        Each dict contains { 'fraction': float, 'accuracy': float, 'roc_auc': float,
                             'feature_importances': dict,
                             'distance_from_prev': float or None,
                             'distance_from_first': float }
    """
    results = []
    feature_names = X.columns.tolist()

    # Split into full training and test set
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    n_train = X_train_full.shape[0]

    cum_frac = 0
    prev_importances = None
    first_importances = None

    for frac in fractions:
        cum_frac += frac
        subset_size = int(n_train * cum_frac)

        X_sub, y_sub = X_train_full[:subset_size], y_train_full[:subset_size]

        print(f"\nTraining on {subset_size} samples ({cum_frac:.2f} fraction)")

        # Train using the optimized function
        model, y_pred_proba, y_pred, optimal_threshold, metrics = train_xgboost_optimal(
            X_sub, X_test, y_sub, y_test
        )

        # Get feature importances
        importances_array = model.feature_importances_
        importances = dict(zip(feature_names, importances_array))

        # Distance from previous model
        if prev_importances is None:
            dist_prev = None
        else:
            prev_arr = np.array(list(prev_importances.values()))
            curr_arr = np.array(list(importances.values()))
            dist_prev = np.linalg.norm(curr_arr - prev_arr)

        # Distance from first model
        if first_importances is None:
            dist_first = 0.0
            first_importances = importances
        else:
            first_arr = np.array(list(first_importances.values()))
            curr_arr = np.array(list(importances.values()))
            dist_first = np.linalg.norm(curr_arr - first_arr)

        results.append({
            'fraction': cum_frac,
            'accuracy': metrics['accuracy'],
            'roc_auc': metrics['roc_auc'],
            'feature_importances': importances,
            'distance_from_prev': dist_prev,
            'distance_from_first': dist_first,
            'optimal_threshold': optimal_threshold
        })

        prev_importances = importances

    return results


def plot_stability_results(results):
    """
    Plots feature importance evolution and distances from stability_training results.
    """
    fractions = [r['fraction'] for r in results]
    feature_names = list(results[0]['feature_importances'].keys())

    # Build DataFrame of feature importances
    fi_df = pd.DataFrame(
        [{f: r['feature_importances'][f] for f in feature_names} for r in results],
        index=fractions
    )

    # Compute per-feature distances (absolute difference from previous step)
    fi_diff_df = fi_df.diff().abs()

    # Distances
    global_dist_prev = [r['distance_from_prev'] for r in results]
    global_dist_first = [r['distance_from_first'] for r in results]
    auc_values = [r['roc_auc'] for r in results]

    # --- Plot Feature Importance Evolution ---
    plt.figure(figsize=(12, 5))
    for feature in feature_names:
        plt.plot(fractions, fi_df[feature], marker='o', label=feature)
    plt.xlabel('Fraction of Training Data')
    plt.ylabel('Feature Importance')
    plt.title('Feature Importance Evolution')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # --- Plot Per-Feature Importance Distance Evolution ---
    plt.figure(figsize=(12, 5))
    for feature in feature_names:
        plt.plot(fractions, fi_diff_df[feature], marker='o', label=feature)
    plt.xlabel('Fraction of Training Data')
    plt.ylabel('Distance from Previous Importance')
    plt.title('Per-Feature Importance Distance Evolution')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # --- Plot Global Distance from Previous + AUC ---
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color = 'tab:red'
    ax1.set_xlabel('Fraction of Training Data')
    ax1.set_ylabel('Distance from Previous', color=color)
    ax1.plot(fractions, global_dist_prev, marker='o', color=color, label='Dist. Prev')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('ROC-AUC', color=color)
    ax2.plot(fractions, auc_values, marker='s', color=color, label='AUC')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 1)

    fig.tight_layout()
    plt.title('Global Distance (Prev) and AUC Evolution')
    fig.legend(loc='upper left', bbox_to_anchor=(0.15, 0.9))
    plt.show()

    # --- Plot Global Distance from First + AUC ---
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color = 'tab:green'
    ax1.set_xlabel('Fraction of Training Data')
    ax1.set_ylabel('Distance from First', color=color)
    ax1.plot(fractions, global_dist_first, marker='o', color=color, label='Dist. First')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('ROC-AUC', color=color)
    ax2.plot(fractions, auc_values, marker='s', color=color, label='AUC')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 1)

    fig.tight_layout()
    plt.title('Global Distance (First) and AUC Evolution')
    fig.legend(loc='upper left', bbox_to_anchor=(0.15, 0.9))
    plt.show()

    # --- Top 5 Features Contributing to Distance (Overall) ---
    feature_cumulative_change = fi_diff_df.sum()
    top5_features = feature_cumulative_change.nlargest(5).index

    plt.figure(figsize=(12, 5))
    for feature in top5_features:
        plt.plot(fractions, fi_diff_df[feature], marker='o', label=feature)

    plt.xlabel('Fraction of Training Data')
    plt.ylabel('Contribution to Distance')
    plt.title('Top 5 Features Contributing to Distance Overall')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
