"""
Step 8: Performance interpretability: Implement the Permutation Importance method to 
identify the main drivers of the predictive performance of your model. Are the drivers of the 
performance metric (Step 8) similar to the drivers of the individual forecasts identified by 
SHAP (Step 7).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def PI(df, model, features, pm, y_true):
    res = []
    y_pred = model.predict_proba(df)[:, 1]
    pm_model = pm(y_true, y_pred)
    n = len(features)
    lim = 3
    for i, feat in enumerate(features):
        if i>=lim:
            break
        temp_df = df.copy()
        col = temp_df[feat]
        col_shuffled = col.sample(frac=1, random_state=42).reset_index(drop=True)
        temp_df[feat] = col_shuffled
        y_pred = model.predict_proba(temp_df)[:, 1]
        pm_shuffled = pm(y_true, y_pred)
        importance = pm_shuffled - pm_model
        res.append(importance)
        print(f"Processed feature {i}/{n}: {feat}, Importance: {importance}")
    if lim<n:
        return res, features[:lim]
    return res, features


def PI_plot(df, model, features, pm, y_true):
    importances, new_features = PI(df, model, features, pm, y_true)
    features = new_features
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    importance_df = importance_df.head(10)
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    plt.title('Permutation Importance of Features')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()
