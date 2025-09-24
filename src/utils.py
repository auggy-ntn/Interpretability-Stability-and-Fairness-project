import pickle
from typing import List, Tuple, Union




def unpack_model(model_name: str) -> object:
    """
    Unpack a serialized model from a pickle file.
    Need to call an abslute path to the model.
    """
    model_path = f"models/{model_name}.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model