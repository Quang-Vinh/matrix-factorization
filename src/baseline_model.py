import numba as nb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from .utils import preprocess_data


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
):
    for epoch in range(n_epochs):
        for i in range(X.shape[0]):
            user, item, rating = int(X[i, 0]), int(X[i, 1]), X[i, 2]

            # Compute error
            rating_pred = global_mean + user_biases[user] + item_biases[item]
            error = rating - rating_pred

            # Update parameters
            user_biases[user] += lr * (error - reg * user_biases[user])
            item_biases[item] += lr * (error - reg * item_biases[item])

        # Display fitting messages
        if verbose == 1:
            rmse = error ** 2
            print("Epoch ", epoch + 1, "/", n_epochs, " -  train_rmse:", rmse)

    return user_biases, item_biases


@nb.njit()
def _predict(
    X: np.ndarray,
    global_mean: float,
    min_rating: int,
    max_rating: int,
    user_biases: np.ndarray,
    item_biases: np.ndarray,
):
    predictions = []

    for i in range(X.shape[0]):
        user, item = int(X[i, 0]), int(X[i, 1])

        if user == -1 or item == -1:
            pred = None
        else:
            pred = global_mean + user_biases[user] + item_biases[item]
            
            # Bound ratings to min and max rating range
            if pred > max_rating:
                pred = max_rating
            elif pred < min_rating:
                pred = min_rating

        predictions.append(pred)

    return predictions


class BaselineModel(BaseEstimator):
    def __init__(
        self,
        n_epochs: int = 100,
        reg: float = 0.02,
        lr: float = 0.005,
        min_rating: int = 0,
        max_rating: int = 5,
        verbose=1,
    ):
        self.n_epochs = n_epochs
        self.reg = reg
        self.lr = lr
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.verbose = verbose
        self.n_users, self.n_items = None, None
        self.global_mean = None
        self.user_biases, self.item_biases = None, None
        self._user_id_map, self._item_id_map = None, None
        return

    def fit(self, X: pd.DataFrame):
        X, self._user_id_map, self._item_id_map = preprocess_data(X)

        self.n_users = len(self._user_id_map)
        self.n_items = len(self._item_id_map)

        # Initialize parameters
        self.user_biases = np.zeros(self.n_users)
        self.item_biases = np.zeros(self.n_items)

        self.global_mean = X["rating"].mean()

        # Run stochastic gradient descent
        user_biases, item_biases = _sgd(
            X=X.to_numpy(),
            global_mean=self.global_mean,
            user_biases=self.user_biases,
            item_biases=self.item_biases,
            n_epochs=self.n_epochs,
            lr=self.lr,
            reg=self.reg,
            verbose=self.verbose,
        )

        return self

    def predict(self):
        # Keep only required columns in given order
        X = X.loc[:, ["u_id", "i_id"]]

        # Remap user_id and item_id
        X.loc[:, "u_id"] = X["u_id"].map(self._user_id_map)
        X.loc[:, "i_id"] = X["i_id"].map(self._item_id_map)

        # Replace missing mappings with -1
        X.fillna(-1, inplace = True)

        # Get predictions
        rating_pred = _predict(X = X.to_numpy, )

        return rating_pred
