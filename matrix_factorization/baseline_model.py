import numba as nb
import numpy as np
import pandas as pd

from .recommender_base import RecommenderBase

from typing import Tuple


class BaselineModel(RecommenderBase):
    """
    Simple model which models the user item rating as r_{ui} = \mu + ubias_u + ibias_i which is sum of a global mean and the corresponding
    user and item biases. The global mean \mu is estimated as the mean of all ratings. The other parameters to be estimated ubias and ibias 
    are vectors of length n_users and n_items respectively. These two vectors are estimated using stochastic gradient descent on the RMSE 
    with regularization.

    NOTE: Recommend method with this model will simply recommend the most popular items for every user. This model should mainly be used
          for estimating the explicit rating for a given user and item 

    Arguments:
        method: {str} -- Method to estimate parameters. Can be one of 'sgd' or 'als' (default: {'sgd'})
        n_epochs {int} -- Number of epochs to train for (default: {100})
        reg {float} -- Lambda parameter for L2 regularization (default: {1})
        lr {float} -- Learning rate for gradient optimization step (default: {0.01})
        min_rating {int} -- Smallest rating possible (default: {0})
        max_rating {int} -- Largest rating possible (default: {5})
        verbose {str} -- Verbosity when fitting. 0 to not print anything, 1 to print fitting model (default: {1})

    Attributes:
        n_users {int} -- Number of users
        n_items {int} -- Number of items
        global_mean {float} -- Global mean of all ratings
        user_biases {numpy array} -- User bias vector of shape (n_users, 1)
        item_biases {numpy array} -- Item bias vector of shape (n_items, i)
        user_id_map {dict} -- Mapping of user ids to assigned integer ids
        item_id_map {dict} -- Mapping of item ids to assigned integer ids
        train_rmse {list} -- Training rmse values
        predictions_possible {list} -- Boolean vector of whether both user and item were known for prediction. Only available after calling predict
    """

    def __init__(
        self,
        method: str = "sgd",
        n_epochs: int = 100,
        reg: float = 1,
        lr: float = 0.01,
        min_rating: int = 0,
        max_rating: int = 5,
        verbose=1,
    ):
        # Check inputs
        if method not in ("sgd", "als"):
            raise ValueError('Method param must be either "sgd" or "als"')

        super().__init__(min_rating=min_rating, max_rating=max_rating, verbose=verbose)

        self.method = method
        self.n_epochs = n_epochs
        self.reg = reg
        self.lr = lr
        return

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """ 
        Fits simple mean and bias model to given user item ratings

        Arguments:
            X {pandas DataFrame} -- Dataframe containing columns user_id, item_id
            y {pandas Series} -- Series containing rating
        """
        X = self._preprocess_data(X=X, y=y, type="fit")
        self.global_mean = X["rating"].mean()

        # Initialize parameters
        self.user_biases = np.zeros(self.n_users)
        self.item_biases = np.zeros(self.n_items)

        # Run parameter estimation
        if self.method == "sgd":
            self.user_biases, self.item_biases, self.train_rmse = _sgd(
                X=X.to_numpy(),
                global_mean=self.global_mean,
                user_biases=self.user_biases,
                item_biases=self.item_biases,
                n_epochs=self.n_epochs,
                lr=self.lr,
                reg=self.reg,
                verbose=self.verbose,
            )

        elif self.method == "als":
            self.user_biases, self.item_biases, self.train_rmse = _als(
                X=X.to_numpy(),
                global_mean=self.global_mean,
                user_biases=self.user_biases,
                item_biases=self.item_biases,
                n_epochs=self.n_epochs,
                reg=self.reg,
                verbose=self.verbose,
            )

        return self

    def predict(self, X: pd.DataFrame, bound_ratings: bool = True) -> list:
        """
        Predict ratings for given users and items

        Arguments:
            X {pd.DataFrame} -- Dataframe containing columns user_id and item_id
            bound_ratings (bool): Whether to bound ratings in range [min_rating, max_rating] (default: True)

        Returns:
            predictions [list] -- List containing rating predictions of all user, items in same order as input X
        """
        # If empty return empty list
        if X.shape[0] == 0:
            return []

        X = self._preprocess_data(X=X, type="predict")

        # Get predictions
        predictions, predictions_possible = _predict(
            X=X.to_numpy(),
            global_mean=self.global_mean,
            min_rating=self.min_rating,
            max_rating=self.max_rating,
            user_biases=self.user_biases,
            item_biases=self.item_biases,
            bound_ratings=bound_ratings,
        )

        self.predictions_possible = predictions_possible

        return predictions

    def update_users(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        lr: float = 0.01,
        n_epochs: int = 20,
        verbose: int = 0,
    ):
        """
        Update user biases vector with new/updated user-item ratings information using SGD. Only the user parameters corresponding for the
        new/updated users will be updated and item parameters will be left alone. 
        
        Note: If updating old users then pass all user-item ratings for old users and not just modified ratings

        Args:
            X (pd.DataFrame): Dataframe containing columns user_id, item_id 
            y (pd.Series): Series containing rating
            lr (float, optional): Learning rate alpha for gradient optimization step
            n_epochs (int, optional): Number of epochs to run SGD. Defaults to 20.
            verbose (int, optional): Verbosity when updating, 0 for nothing and 1 for training messages. Defaults to 0.
        """
        X, known_users, new_users = self._preprocess_data(X=X, y=y, type="update")

        # Re-initialize user bias for old users
        for user in known_users:
            user_index = self.user_id_map[user]
            self.user_biases[user_index] = 0

        # Add user bias param for new users
        self.user_biases = np.append(self.user_biases, np.zeros(len(new_users)))

        # Estimate new bias parameter
        self.user_biases, _, self.train_rmse = _sgd(
            X=X.to_numpy(),
            global_mean=self.global_mean,
            user_biases=self.user_biases,
            item_biases=self.item_biases,
            n_epochs=n_epochs,
            lr=lr,
            reg=self.reg,
            verbose=verbose,
            update_item_params=False,
        )

        return


@nb.njit()
def _calculate_rmse(
    X: np.ndarray, global_mean: float, user_biases: np.ndarray, item_biases: np.ndarray
):
    """
    Calculates root mean squared error for given data and model parameters

    Args:
        X (np.ndarray): Matrix with columns user, item and rating
        global_mean (float): Global mean rating
        user_biases (np.ndarray): User biases vector of shape (n_users, 1)
        item_biases (np.ndarray): Item biases vector of shape (n_items, 1)

    Returns:
        rmse [float]: Root mean squared error
    """
    n_ratings = X.shape[0]
    errors = np.zeros(n_ratings)

    # Iterate through all user-item ratings
    for i in range(n_ratings):
        user_id, item_id, rating = int(X[i, 0]), int(X[i, 1]), X[i, 2]

        # Calculate prediction and error
        pred = global_mean + user_biases[user_id] + item_biases[item_id]
        errors[i] = rating - pred

    rmse = np.sqrt(np.square(errors).mean())

    return rmse


@nb.njit()
def _sgd(
    X: np.ndarray,
    global_mean: float,
    user_biases: np.ndarray,
    item_biases: np.ndarray,
    n_epochs: int,
    lr: float,
    reg: float,
    verbose: int,
    update_user_params: bool = True,
    update_item_params: bool = True,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Performs Stochastic Gradient Descent to estimate the user_biases and item_biases

    Arguments:
        X {numpy array} -- User-item rating matrix
        global_mean {float} -- Global mean of all ratings
        user_biases {numpy array} -- User biases vector of shape (n_users, 1)
        item_biases {numpy array} -- Item biases vector of shape (n_items, 1)
        n_epochs {int} -- Number of epochs to run
        lr {float} -- Learning rate alpha
        reg {float} -- Regularization parameter lambda for Frobenius norm
        verbose {int} -- Verbosity when fitting. 0 for nothing and 1 for printing epochs
        update_user_params {bool} -- Whether to update user bias parameters or not. Default is True.
        update_item_params {bool} -- Whether to update item bias parameters or not. Default is True.

    Returns:
        user_biases [np.ndarray] -- Updated user_biases vector
        item_biases [np.ndarray] -- Updated item_bases vector
        train_rmse -- Training rmse values
    """
    train_rmse = []

    for epoch in range(n_epochs):
        # Shuffle data before each epoch
        np.random.shuffle(X)

        # Iterate through all user-item ratings
        for i in range(X.shape[0]):
            user_id, item_id, rating = int(X[i, 0]), int(X[i, 1]), X[i, 2]

            # Compute error
            rating_pred = global_mean + user_biases[user_id] + item_biases[item_id]
            error = rating - rating_pred

            # Update parameters
            if update_user_params:
                user_biases[user_id] += lr * (error - reg * user_biases[user_id])
            if update_item_params:
                item_biases[item_id] += lr * (error - reg * item_biases[item_id])

        # Calculate error and print
        rmse = _calculate_rmse(
            X=X,
            global_mean=global_mean,
            user_biases=user_biases,
            item_biases=item_biases,
        )
        train_rmse.append(rmse)

        if verbose == 1:
            print("Epoch ", epoch + 1, "/", n_epochs, " -  train_rmse:", rmse)

    return user_biases, item_biases, train_rmse


@nb.njit()
def _als(
    X: np.ndarray,
    global_mean: float,
    user_biases: np.ndarray,
    item_biases: np.ndarray,
    n_epochs: int,
    reg: float,
    verbose: int,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Performs Alternating Least Squares to estimate the user_biases and item_biases. For every epoch, the item biases are held constant while
    solving directly for the user biases parameters using a closed form equation. Then the user biases parameters is held constant and the same
    is done for the item biases. This can be derived easily and is given in the lecture here https://www.youtube.com/watch?v=gCaOa3W9kM0&t=32m55s
    which is also similar to the implementation in Surprise.

    Arguments:
        X {numpy array} -- User-item rating matrix
        global_mean {float} -- Global mean of all ratings
        user_biases {numpy array} -- User biases vector of shape (n_users, 1)
        item_biases {numpy array} -- Item biases vector of shape (n_items, 1)
        n_epochs {int} -- Number of epochs to run
        reg {float} -- Regularization parameter lambda for Frobenius norm
        verbose {int} -- Verbosity when fitting. 0 for nothing and 1 for printing epochs

    Returns:
        user_biases [np.ndarray] -- Updated user_biases vector
        item_biases [np.ndarray] -- Updated item_bases vector
        train_rmse -- Training rmse values
    """
    n_users = user_biases.shape[0]
    n_items = item_biases.shape[0]
    train_rmse = []

    # Get counts of all users and items
    user_counts = np.zeros(n_users)
    item_counts = np.zeros(n_items)
    for i in range(X.shape[0]):
        user_id, item_id = int(X[i, 0]), int(X[i, 1])
        user_counts[user_id] += 1
        item_counts[item_id] += 1

    # For each epoch optimize User biases, and then Item biases
    for epoch in range(n_epochs):

        # Update user bias parameters
        user_biases = np.zeros(n_users)

        # Iterate through all user-item ratings
        for i in range(X.shape[0]):
            user_id, item_id, rating = int(X[i, 0]), int(X[i, 1]), X[i, 2]
            user_biases[user_id] += rating - global_mean - item_biases[item_id]

        # Set user bias estimation
        user_biases = user_biases / (reg + user_counts)

        # Update item bias parameters
        item_biases = np.zeros(n_items)

        # Iterate through all user-item ratings
        for i in range(X.shape[0]):
            user_id, item_id, rating = int(X[i, 0]), int(X[i, 1]), X[i, 2]
            item_biases[item_id] += rating - global_mean - user_biases[user_id]

        # Set item bias estimation
        item_biases = item_biases / (reg + item_counts)

        # Calculate error and print
        rmse = _calculate_rmse(
            X=X,
            global_mean=global_mean,
            user_biases=user_biases,
            item_biases=item_biases,
        )
        train_rmse.append(rmse)

        if verbose == 1:
            print("Epoch ", epoch + 1, "/", n_epochs, " -  train_rmse:", rmse)

    return user_biases, item_biases, train_rmse


@nb.njit()
def _predict(
    X: np.ndarray,
    global_mean: float,
    min_rating: int,
    max_rating: int,
    user_biases: np.ndarray,
    item_biases: np.ndarray,
    bound_ratings: bool,
) -> Tuple[list, list]:
    """
    Calculate predicted ratings for each user-item pair.

    Arguments:
        X {np.ndarray} -- Matrix with columns representing (user_id, item_id)
        global_mean {float} -- Global mean of all ratings
        min_rating {int} -- Lowest rating possible
        max_rating {int} -- Highest rating possible
        user_biases {np.ndarray} -- User biases vector of length n_users
        item_biases {np.ndarray} -- Item biases vector of length n_items
        bound_ratings {boolean} -- Whether to bound predictions in between range [min_rating, max_rating]

    Returns:
        predictions [np.ndarray] -- Vector containing rating predictions of all user, items in same order as input X
        predictions_possible [np.ndarray] -- Vector of whether both given user and item were contained in the data that the model was fitted on
    """

    predictions = []
    predictions_possible = []

    for i in range(X.shape[0]):
        user_id, item_id = int(X[i, 0]), int(X[i, 1])
        user_known = user_id != -1
        item_known = item_id != -1

        rating_pred = global_mean

        if user_known:
            rating_pred += user_biases[user_id]
        if item_known:
            rating_pred += item_biases[item_id]

        # Bound ratings to min and max rating range
        if bound_ratings:
            if rating_pred > max_rating:
                rating_pred = max_rating
            elif rating_pred < min_rating:
                rating_pred = min_rating

        predictions.append(rating_pred)
        predictions_possible.append(user_known and item_known)

    return predictions, predictions_possible
