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
    min_val = feature_values.min()
    max_val = feature_values.max()
    return pd.Series([min_val + i * (max_val - min_val) / (num_points - 1) for i in range(num_points)])

def fairness_partial_dependance_plots(df, model, protected_attribute, n_points=10, file_dir=None, threshold=.05, categorical_features=[]):
    """Generate a partial dependence plot for a given feature."""

    df_categorical_columns = { 
        cat_feature: [
            col for col in df.columns if col.startswith(cat_feature)
        ] for cat_feature in categorical_features 
    }
    categorical_columns = [
        col for cols in df_categorical_columns.values() for col in cols
    ]
    df_continuous_columns = [
        col for col in df.columns if col not in categorical_columns
    ]

    for categorical_feature, values in df_categorical_columns.items():
        df_temp = df.copy()
        p_values = []
        labels = []
        for value in values:
            df_temp[value] = True
            for other_value in values:
                if other_value != value:
                    df_temp[other_value] = False
            p_value = fairness_test_statistic(df_temp, model, protected_attribute)
            p_values.append(p_value)
            labels.append(value)

        plt.figure()
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.title(f'Partial Dependence Plot for {categorical_feature}')
        plt.bar(labels, p_values, align='center')
        plt.xticks(rotation=45)
        if max(p_values) >= threshold:
            plt.axhline(y=threshold, color='r', linestyle='--', label='Significance Threshold')

        plt.xlabel(categorical_feature)
        plt.ylabel('Fairness Test Statistic (p-value)')
        plt.grid()
        if file_dir:
            plt.tight_layout()
            plt.savefig(f'{file_dir}/partial_dependence_{categorical_feature}.png')
        if max(p_values) >= threshold:
            plt.show()

    for feature in df_continuous_columns:
        df_temp = df.copy()
        points = get_points_for_partial_dependance(df, feature, n_points)
        p_values = []
        for point in points:
            df_temp[feature] = point
            p_value = fairness_test_statistic(df_temp, model, protected_attribute)
            p_values.append(p_value)

        plt.figure()
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.title(f'Partial Dependence Plot for {feature}')
        plt.plot(points, p_values)
        if max(p_values) >= threshold:
            plt.axhline(y=threshold, color='r', linestyle='--', label='Significance Threshold')

        plt.xlabel(feature)
        plt.ylabel('Fairness Test Statistic (p-value)')
        plt.grid()
        if file_dir:
            plt.savefig(f'{file_dir}/partial_dependence_{feature}.png')
        if max(p_values) >= threshold:
            plt.show()
