import numpy as np
import pandas as pd


def preprocess_data(X: pd.DataFrame) -> (pd.DataFrame, dict, dict):
    """
    Preprocesses user-item ratings dataframe before fitting

    Arguments:
        X {pd.DataFrame} -- Dataframe containing columns user_id, item_id and rating

    Returns:
        X [pd.DataFrame] -- Dataframe with columns user_id, item_id and rating
        user_id_map [dict] -- Dictionary containing mapping of user_ids to assigned integer ids 
        item_id_map [dict] -- Dictionary containing mapping of item_ids to assigned integer ids
    """
    # Keep only required columns in given order
    X = X.loc[:, ["user_id", "item_id", "rating"]]

    # Check for duplicate user-item ratings
    if X.duplicated(subset=["user_id", "item_id"]).sum() != 0:
        raise Exception("Duplicate user-item ratings in matrix")

    # Map user_id and item_id to integers
    user_ids = X["user_id"].unique()
    item_ids = X["item_id"].unique()
    user_id_map = {user_id: i for (i, user_id) in enumerate(user_ids)}
    item_id_map = {item_id: i for (i, item_id) in enumerate(item_ids)}

    # Remap user id and item id to assigned integer ids
    X.loc[:, "user_id"] = X["user_id"].map(user_id_map)
    X.loc[:, "item_id"] = X["item_id"].map(item_id_map)

    return X, user_id_map, item_id_map


def preprocess_data_update(
    X: pd.DataFrame, item_id_map: dict
) -> (pd.DataFrame, dict, dict):
    """
    Preprocesses user-item ratings dataframe of new or updated ratings

    Args:
        X (pd.DataFrame): Dataframe with columns user_id, item_id and rating 
        item_id_map (dict): Dictionary containing mapping of item_ids to assigned integer ids

    Returns:
        X [pd.DataFrame]: Dataframe with only columns user_id, item_id and rating
    """
    # Keep only item and ratings
    X = X.loc[:, ["user_id", "item_id", "rating"]]

    # If there are duplicates then stop
    if X.duplicated().sum() != 0:
        raise Exception("Duplicate items in matrix")

    # Keep only item ratings for which the item is known
    items = item_id_map.keys()
    X = X.query("item_id in @items").copy()

    return X


def preprocess_data_predict(
    X: pd.DataFrame, user_id_map: dict, item_id_map: dict
) -> pd.DataFrame:
    """
    Preprocess user-item dataframe for prediction

    Args:
        X (pd.DataFrame): Dataframe with columns user_id and item_id
        user_id_map (dict): Dictionary containing mapping of user_ids to assigned integer ids
        item_id_map (dict): Dictionary containing mapping of item_ids to assigned integer ids

    Returns:
        pd.DataFrame: [description]
    """
    # Keep only required columns in given order
    X = X.loc[:, ["user_id", "item_id"]]

    # Remap user_id and item_id
    X.loc[:, "user_id"] = X["user_id"].map(user_id_map)
    X.loc[:, "item_id"] = X["item_id"].map(item_id_map)

    # Replace missing mappings with -1
    X.fillna(-1, inplace=True)

    return X
