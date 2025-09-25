import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

import constants.column_names as cst


def get_preprocessed_data(
    path: str = r"../data/dataproject2025.csv",
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
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
    # Load dataset
    df = pd.read_csv(path)

    # Drop unnecessary columns
    df = df.drop(columns=cst.DROP_COLUMNS)

    # Handle missing values
    df = df.replace("nan", np.nan)
    df.dropna(inplace=True)

    # Separate features, predicted probabilities, predictions, and true labels
    prob = df.pop("Predicted probabilities")
    predictions = df.pop("Predictions")
    true_labels = df.pop("target")

    # One-hot encode categorical features
    encoder = OneHotEncoder(sparse_output=False, drop="first")
    encoded_cols = encoder.fit_transform(df[cst.CATEGORICAL_COLUMNS])
    encoded_col_names = encoder.get_feature_names_out(cst.CATEGORICAL_COLUMNS)
    df_encoded = pd.DataFrame(encoded_cols, columns=encoded_col_names, index=df.index)

    # Ordinal encode ordinal features
    ordinal_encoder = OrdinalEncoder()
    df[cst.ORDINAL_COLUMNS] = ordinal_encoder.fit_transform(df[cst.ORDINAL_COLUMNS])

    # Combine encoded features with the rest of the data
    df = pd.concat([df.drop(columns=cst.CATEGORICAL_COLUMNS), df_encoded], axis=1)

    # Standardize numerical features
    binary_col_names = cst.BINARY_COLUMNS + encoded_col_names.tolist()
    binary_cols = df[binary_col_names]
    continuous_cols = df.select_dtypes(include=np.number).drop(
        columns=binary_col_names
    )
    scaler = StandardScaler()
    scaled_continuous = pd.DataFrame(
        scaler.fit_transform(continuous_cols),
        columns=continuous_cols.columns,
        index=df.index,
    )

    # Combine scaled continuous features with binary features
    df = pd.concat([scaled_continuous, binary_cols], axis=1)

    # Rename columns
    df = df.rename(columns=cst.RENAMING_DICT)

    return df, prob, predictions, true_labels