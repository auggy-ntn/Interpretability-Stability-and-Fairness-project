import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def get_preprocessed_data(path=None) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Preprocesses the dataset by:
    - Dropping unnecessary columns and handling missing values.
    - Separating predicted probabilities, predictions, and true labels.
    - One-hot encoding categorical features.
    - Ordinal encoding ordinal features.
    - Standardizing continuous features while keeping binary features unchanged.

    Args:
        path (str): Path to the dataset CSV file. Defaults to "../data/dataproject2025.csv".

    Returns:
        tuple: A tuple containing:
            - Preprocessed DataFrame (pd.DataFrame)
            - Predicted probabilities (pd.Series)
            - Predictions (pd.Series)
            - True labels (pd.Series)
    """
    path = r"../data/dataproject2025.csv"
    df = pd.read_csv(path)

    # Drop unnecessary columns
    df = df.drop(columns=['Unnamed: 0', 'sub_grade'])

    # Handle missing values
    df = df.replace('nan', np.nan)
    df.dropna(inplace=True)

    # Separate features, predicted probabilities, predictions, and true labels
    prob = df.pop('Predicted probabilities')
    predictions = df.pop('Predictions')
    true_labels = df.pop('target')

    # Define categorical and ordinal features
    categorical_features = ['purpose', 'home_ownership', 'emp_title', 'emp_length']
    ordinal_features = ['grade']

    # One-hot encode categorical features
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_cols = encoder.fit_transform(df[categorical_features])
    encoded_col_names = encoder.get_feature_names_out(categorical_features)
    df_encoded = pd.DataFrame(encoded_cols, columns=encoded_col_names, index=df.index)

    # Ordinal encode ordinal features
    ordinal_encoder = OrdinalEncoder()
    df[ordinal_features] = ordinal_encoder.fit_transform(df[ordinal_features])

    # Combine encoded features with the rest of the data
    df = pd.concat([df.drop(columns=categorical_features), df_encoded], axis=1)

    # Standardize numerical features
    binary_cols = df.loc[:, df.nunique() == 2]
    continuous_cols = df.loc[:, df.nunique() > 2]
    scaler = StandardScaler()
    scaled_continuous = pd.DataFrame(scaler.fit_transform(continuous_cols), columns=continuous_cols.columns, index=df.index)

    # Combine scaled continuous features with binary features
    df = pd.concat([scaled_continuous, binary_cols], axis=1)

    return df, prob, predictions, true_labels