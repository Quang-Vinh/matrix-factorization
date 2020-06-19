import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def train_update_test_split(X: pd.DataFrame, frac_new_users: float):
    users = X["user_id"].unique()

    # Users that won't be included in the initial training
    users_update = np.random.choice(
        users, size=round(frac_new_users * len(users)), replace=False
    )

    # Initial training matrix
    train_initial = X.query("user_id not in @users_update")

    # Train and test sets for updating model. For each new user split their ratings into two sets, one for update and one for test
    data_update = X.query("user_id in @users_update")
    train_update, test_update = train_test_split(
        data_update, stratify=data_update["user_id"], test_size=0.5
    )

    return train_initial, train_update, test_update

