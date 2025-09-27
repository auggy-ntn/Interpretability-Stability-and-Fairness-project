from scipy.stats import chi2_contingency
import pandas as pd
import matplotlib.pyplot as plt

def statistical_parity(df, feature, target):
    """Calculate the Statistical Parity (SP) for a given feature."""
    # Group by the feature and calculate the mean of the target
    group_means = df.groupby(feature)[target].mean()
    
    # Calculate the absolute difference in means between groups
    sp_value = abs(group_means.max() - group_means.min())
    
    return sp_value

def fairness_test_statistic(df, model, protected_attribute):
    """Calculate the fairness test statistic (SP or CSP) for a given feature."""
    preds = model.predict(df)
    contingency_table = pd.crosstab(protected_attribute, preds)
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    return p

def get_points_for_partial_dependance(df, feature, num_points=10):
    """Get points for partial dependence plot."""
    feature_values = df[feature]
    if feature_values.nunique() <= num_points:
        return feature_values.unique(), 'discrete'
    min_val = feature_values.min()
    max_val = feature_values.max()
    return pd.Series([min_val + i * (max_val - min_val) / (num_points - 1) for i in range(num_points)]), 'continuous'

def fairness_partial_dependance_plot(df, model, feature, protected_attribute, n_points=10, filename=None):
    """Generate a partial dependence plot for a given feature."""

    df_temp = df.copy()
    points, mode = get_points_for_partial_dependance(df, feature, n_points)
    p_values = []
    for point in points:
        df_temp[feature] = point
        p_value = fairness_test_statistic(df_temp, model, protected_attribute)
        p_values.append(p_value)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.title(f'Partial Dependence Plot for {feature}')
    if mode == 'continuous':
        plt.plot(points, p_values)
    else:
        plt.bar(points, p_values, align='center')

    plt.xlabel(feature)
    plt.ylabel('Fairness Test Statistic (p-value)')
    plt.grid()
    if filename:
        plt.savefig(filename)
    plt.show()

def fairness_partial_dependance_plots(df, model, protected_attribute, n_points=10, file_dir=None, threshold=.05):
    """Generate a partial dependence plot for a given feature."""

    for feature in df.columns:
        df_temp = df.copy()
        points, mode = get_points_for_partial_dependance(df, feature, n_points)
        p_values = []
        for point in points:
            df_temp[feature] = point
            p_value = fairness_test_statistic(df_temp, model, protected_attribute)
            p_values.append(p_value)

        plt.figure()
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.title(f'Partial Dependence Plot for {feature}')
        if mode == 'continuous':
            plt.plot(points, p_values)
        else:
            plt.bar(points, p_values, align='center')
        if max(p_values) >= threshold:
            plt.plot([points.min(), points.max()], [threshold, threshold], 'r--', label='Significance Threshold')

        plt.xlabel(feature)
        plt.ylabel('Fairness Test Statistic (p-value)')
        plt.grid()
        if file_dir:
            plt.savefig(f'{file_dir}/partial_dependence_{feature}.png')
        if max(p_values) >= threshold:
            plt.show()
