import pickle
from typing import List, Tuple, Union
import numpy as np

def unpack_model(model_path: str) -> object:
    """
    Unpack a serialized model from a pickle file.
    Need to call an absolute path to the model.
    """
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def quantile_binary(column, p):
    proportion = p / 100.0 if p > 1 else p
    threshold = column.quantile(1 - proportion, interpolation="linear")
    flags = (column >= threshold).astype(int)
    return flags

def random_binary(column):
    ps = column / 100.0 if column.max() > 1 else column
    flags = np.random.random_sample(column.shape) < ps
    return flags

def over_pct_binary(column, p=.5):
    ps = column / 100.0 if column.max() > 1 else column
    p = p / 100.0 if p > 1 else p
    flags = (ps >= p).astype(int)
    return flags