import pickle
from typing import List, Tuple, Union


def unpack_model(model_name: str) -> object:
    """
    Unpack a serialized model from a pickle file.
    Need to call an absolute path to the model.
    """
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model
