import pandas as pd

from sklearn.preprocessing import OneHotEncoder






def one_hot(df, categorical_features):
    copy = df.copy()
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_cols = encoder.fit_transform(copy[categorical_features])
    encoded_col_names = encoder.get_feature_names_out(categorical_features)
    encoded = pd.DataFrame(encoded_cols, columns=encoded_col_names)
    res = pd.concat([copy.drop(columns=categorical_features).reset_index(drop=True), encoded.reset_index(drop=True)], axis=1)
    return res



