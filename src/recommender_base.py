import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


class RecommenderBase(BaseEstimator):
    """
    Abstract base class for all recommender models.
    All subclasses should implement the fit() and predict() methods
    """

    def __init__(self):
        return

    def fit(self, X: pd.DataFrame):
        """
        Fit model to given data

        Args:
            X {pandas DataFrame} -- Dataframe containing columns user_id, item_id and rating
        """
        return self

    def predict(self, X: pd.DataFrame) -> list:
        """
        Predict ratings for given users and items

        Args:
            X (pd.DataFrame): Dataframe containing columns user_id and item_id

        Returns:
            list: List containing rating predictions of all user, items in same order as input X
        """
        return []

    def recommend(
        self,
        user,
        amount: int = 10,
        items_known: list = None,
        include_user: bool = True,
    ) -> pd.DataFrame:
        """
        Returns a DataFrame of recommendations of items for a given user sorted from highest to lowest.

        Args:
            user (any): User_id to get recommendations for (not assigned user_id from self.user_id_map)
            items_known (list, optional): List of items already known by user and to not be considered in recommendations. Defaults to None.
            include_user (bool, optional): Whether to include the user_id in the output DataFrame or not. Defaults to True.

        Returns:
            pd.DataFrame: Recommendations DataFrame for user with columns user_id (optional), item_id, rating sorted from highest to lowest rating 
        """
        items = list(self.item_id_map.keys())

        # If items_known is provided then filter by items that the user does not know
        if items_known is not None:
            items_known = list(items_known)
            items = [item for item in items if item not in items_known]

        # Get rating predictions for given user and all unknown items
        items_recommend = pd.DataFrame({"user_id": user, "item_id": items})
        items_recommend["rating_pred"] = self.predict(X=items_recommend)

        # Sort and keep top n items
        items_recommend.sort_values(by="rating_pred", ascending=False, inplace=True)
        items_recommend = items_recommend.head(amount)

        if not include_user:
            items_recommend.drop(["user_id"], axis="columns", inplace=True)

        return items_recommend

