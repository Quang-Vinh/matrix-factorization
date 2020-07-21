import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from typing import Tuple


def train_update_test_split(
    X: pd.DataFrame, frac_new_users: float
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Split data into 3 parts (train_initial, train_update, test_update) for testing performance of model update for new users. First, a set of new
    users is set and all ratings corresponding to all other users is assigned to train_initial. Then, for each new user half of their ratings are
    stored in train_update and half are stored in test_update. 

    To use the three sets returned:
        1. Fit your model to train_update set.
        2. Update your model with train_update 
        3. Calculate predictions on test_update and compare with their actual ratings

    Args:
        X (pd.DataFrame): Data frame containing columns user_id, item_id
        frac_new_users (float): Fraction of users to not include in train_initial

    Returns:
        X_train_initial [pd.DataFrame]: Training set user_ids and item_ids for initial model fitting
        y_train_initial [pd.Series]: Corresponding ratings for X_train_initial
        X_train_update [pd.DataFrame]: Training set user_ids and item_ids for model updating. Contains users that are not in train_initial
        y_train_initial [pd.Series]: Corresponding ratings for X_train_update
        X_test_update [pd.DataFrame]: Testing set user_ids and item_ids for model updating. Contains same users as train_update
        y_test_update [pd.Series]: Corresponding ratings for X_test_update
    """
    users = X["user_id"].unique()

    # Users that won't be included in the initial training
    users_update = np.random.choice(
        users, size=round(frac_new_users * len(users)), replace=False
    )

    # Initial training matrix
    train_initial = X.query("user_id not in @users_update").sample(
        frac=1, replace=False
    )

    # Train and test sets for updating model. For each new user split their ratings into two sets, one for update and one for test
    data_update = X.query("user_id in @users_update")
    train_update, test_update = train_test_split(
        data_update, stratify=data_update["user_id"], test_size=0.5
    )

    # Split into X and y
    X_train_initial, y_train_initial = (
        train_initial[["user_id", "item_id"]],
        train_initial["rating"],
    )
    X_train_update, y_train_update = (
        train_update[["user_id", "item_id"]],
        train_update["rating"],
    )
    X_test_update, y_test_update = (
        test_update[["user_id", "item_id"]],
        test_update["rating"],
    )

    return (
        X_train_initial,
        y_train_initial,
        X_train_update,
        y_train_update,
        X_test_update,
        y_test_update,
    )

