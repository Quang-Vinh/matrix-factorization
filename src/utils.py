import numpy as np
import pandas as pd


def preprocess_data(X: pd.DataFrame) -> (np.ndarray, dict, dict):
    """Preprocesses ratings dataframe before fitting

    Arguments:
        X {pd.DataFrame} -- Dataframe containing columns u_id, i_id and rating

    Returns:
        X [pd.DataFrame] -- Dataframe with only columns u_id, i_id and rating in that order
        user_id_map [dict] -- Dictionary containing mapping of user_ids to assigned integer ids 
        item_id_map [dict] -- Dictionary containing mapping of item_ids to assigned integer ids
    """
    # Keep only required columns in given order
    X = X.loc[:, ["u_id", "i_id", "rating"]]

    # Map u_id and i_id to integers
    user_ids = X["u_id"].unique()
    item_ids = X["i_id"].unique()
    user_id_map = {u_id: i for (i, u_id) in enumerate(user_ids)}
    item_id_map = {i_id: i for (i, i_id) in enumerate(item_ids)}

    # Remap user id and item id to assigned integer ids
    X.loc[:, "u_id"] = X["u_id"].map(user_id_map)
    X.loc[:, "i_id"] = X["i_id"].map(item_id_map)

    return X, user_id_map, item_id_map
