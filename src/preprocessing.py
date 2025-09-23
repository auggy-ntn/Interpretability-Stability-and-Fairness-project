import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def one_hot(df, categorical_features):
    copy = df.copy()
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_cols = encoder.fit_transform(copy[categorical_features])
    encoded_col_names = encoder.get_feature_names_out(categorical_features)
    encoded = pd.DataFrame(encoded_cols, columns=encoded_col_names)
    res = pd.concat([copy.drop(columns=categorical_features).reset_index(drop=True), encoded.reset_index(drop=True)], axis=1)
    return res

# load data (personal path, needs to be changed)
path = r"../data/dataproject2025.csv"

def get_data():

    df = pd.read_csv(path)

    # rename index column and drop old index column
    df['Index'] = df['Unnamed: 0']
    df = df.drop(columns=['Unnamed: 0'])
    #df.info()
    # number of missing values
    df=df.replace('nan', np.nan)
    df.drop(index=df[df.isna().any(axis=1)].index, inplace=True)
    # check
    # separate features, predicted probabilities, predictions and true labels
    prob = df['Predicted probabilities']
    df.drop(columns=['Predicted probabilities'], inplace=True)
    predictions = df['Predictions']
    df.drop(columns=['Predictions'], inplace=True)
    true_labels = df['target']
    df.drop(columns=['target'], inplace=True)
    # Taking all features into account, one hot encoding
    categorical_features = ['sub_grade', 'purpose', 'home_ownership', 'grade', 'emp_title', 'emp_length']

    df_oh = one_hot(df, categorical_features)
    # df_oh.info()
    return df_oh, prob, predictions, true_labels