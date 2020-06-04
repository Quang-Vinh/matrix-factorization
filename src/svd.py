import numba as nb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from .utils import preprocess_data


# TODO: Add early stopping, and also Minibatch GD as an option
# TODO: Save training RMSE values
# TODO: Add Incremental option
# TODO: Add non-linearity function according to Simon Funk


@nb.njit()
def _sgd(
    X: np.ndarray,
    global_mean: float,
    user_features: np.ndarray,
    item_features: np.ndarray,
    user_biases: np.ndarray,
    item_biases: np.ndarray,
    n_epochs: int,
    lr: float,
    reg: float,
    verbose: int,
):
    """
    Performs stochastic gradient descent. Similar to https://github.com/gbolmier/funk-svd and https://github.com/NicolasHug/Surprise we iterate
    over each factor manually for a given user/item instead of indexing by a row such as user_feature[user] since it has shown to be much faster. 
    We have also tested with representing user_features and item_features as 1D arrays but that also is much slower. Using parallel turned on in numba
    gives much worse performance as well.

    Arguments:
        X {numpy array} -- User-item ranking matrix
        global_mean {float} -- Global mean of all ratings
        user_features {numpy array} -- Start matrix U of user features of shape (n_users, n_factors)
        item_features {numpy array} -- Start matrix V of item features of shape (n_items, n_factors)
        user_biases {numpy array} -- User biases vector of shape (n_users, 1)
        item_biases {numpy array} -- Item biases vector of shape (n_items, 1)
        n_epochs {int} -- Number of epochs to run
        lr {float} -- Learning rate alpha
        reg {float} -- Regularization parameter lambda for L2 norm
        verbose {int} -- Verbosity when fitting. 0 for nothing and 1 for printing epochs

    Returns:
        user_features [np.ndarray] -- Updated user_features matrix U
        item_features [np.ndarray] -- Updated item_features matrix V
        user_biases [np.ndarray] -- Updated user_biases vector
        item_biases [np.ndarray] -- Updated item_bases vector
    """
    n_factors = user_features.shape[1]
    error = 0

    for epoch in range(n_epochs):
        for i in range(X.shape[0]):
            user, item, rating = int(X[i, 0]), int(X[i, 1]), X[i, 2]
            item_bias = item_biases[item]
            user_bias = user_biases[user]

            # Compute error
            rating_pred = global_mean + item_bias + user_bias
            for f in range(n_factors):
                rating_pred += user_features[user, f] * item_features[item, f]
            error = rating - rating_pred

            # Update parameters
            item_biases[item] += lr * (error - reg * item_bias)
            user_biases[user] += lr * (error - reg * user_bias)
            for f in range(n_factors):
                user_feature = user_features[user, f]
                item_feature = item_features[item, f]
                user_features[user, f] += lr * (
                    error * item_feature - reg * user_feature
                )
                item_features[item, f] += lr * (
                    error * user_feature - reg * item_feature
                )

        # Display training message
        if verbose == 1:
            rmse = error ** 2
            print("Epoch ", epoch + 1, "/", n_epochs, " -  train_rmse:", rmse)

    return user_features, item_features, user_biases, item_biases


@nb.njit()
def _predict(
    X: np.ndarray,
    global_mean: float,
    min_rating: int,
    max_rating: int,
    user_features: np.ndarray,
    item_features: np.ndarray,
    user_biases: np.ndarray,
    item_biases: np.ndarray,
):
    """[summary]

    Arguments:
        X {np.ndarray} -- Matrix with columns representing (user_id, item_id)
        global_mean {float} -- Global mean of all ratings
        user_features {np.ndarray} -- User features matrix U of shape (n_users, n_factors)
        item_features {np.ndarray} -- Item features matrix P of shape (n_items, n_factors)
        user_biases {np.ndarray} -- User biases vector of length n_users
        item_biases {np.ndarray} -- Item biases vector of length n_items

    Returns:
        predictions [np.ndarray] -- Vector containing rating predictions of all user, items in same order as input X
        predictions_possible [np.ndarray] -- Vector of whether both given user and item were contained in the data that the model was fitted on
    """
    predictions = []
    predictions_possible = []

    for i in range(X.shape[0]):
        user, item = int(X[i, 0]), int(X[i, 1])
        user_known = user != -1
        item_known = item != -1

        rating_pred = global_mean

        # If known user or time then add corresponding bias and feature vector product
        if user_known:
            rating_pred += user_biases[user]
        if item_known:
            rating_pred += item_biases[item]
        if user_known and item_known:
            rating_pred += np.dot(user_features[user, :], item_features[item, :])

        # Bound ratings to min and max rating range
        if rating_pred > max_rating:
            rating_pred = max_rating
        elif rating_pred < min_rating:
            rating_pred = min_rating

        predictions.append(rating_pred)
        predictions_possible.append(user_known and item_known)

    return predictions, predictions_possible


class SVD(BaseEstimator):
    """ 
    Singular Value Decomposition by Simon Funk. Finds the thin matrices U and V such that U * V^T give a good approximation to the user-item 
    ratings matrix based on rmse. SVD decomposes a matrix into matrices U, S, V^T where S is a diagonal matrix containing the singular values
    however with this algorithm S will be mixed into U and V^T so we simply just end up with 2 matrices instead of 3.

    Arguments:
        n_factors {int} -- The number of latent factors in matrices P and U (default: {100})
        n_epochs {int} -- Number of epochs to train for (default: {100})
        reg {float} -- Lambda parameter for L2 regularization (default: {0.2})
        lr {float} -- Learning rate for gradient optimisation step (default: {0.005})
        init_mean {float} -- Mean of normal distribution to use for initializing parameters (default: {0})
        init_sd {float} -- Standard deviation of normal distribution to use for initializing parameters (default: {0.1})
        min_rating {int} -- Smallest rating possible (default: {0})
        max_rating {int} -- Largest rating possible (default: {5})
        verbose {str} -- Verbosity when fitting. 0 to not print anything, 1 to print fitting model (default: {1})

    Attributes:
        n_users {int} -- Number of users
        n_items {int} -- Number of items
        global_mean {float} -- Global mean of all ratings
        user_features {numpy array} -- Decomposed U matrix of user features of shape (n_users, n_factors)
        item_features {numpy array} -- Decomposed V matrix of item features of shape (n_items, n_factors)
        user_biases {numpy array} -- User bias vector of shape (n_users, 1)
        item_biases {numpy array} -- Item bias vector of shape (n_items, i)
        _user_id_map {dict} -- Mapping of user ids to assigned integer ids
        _item_id_map {dict} -- Mapping of item ids to assigned integer ids
    """

    def __init__(
        self,
        n_factors: int = 100,
        n_epochs: int = 100,
        reg: float = 0.2,
        lr: float = 0.005,
        init_mean: float = 0,
        init_sd: float = 0.1,
        min_rating: int = 0,
        max_rating: int = 5,
        verbose: int = 1,
    ):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.reg = reg
        self.lr = lr
        self.init_mean = init_mean
        self.init_sd = init_sd
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.verbose = verbose
        self.n_users, self.n_items = None, None
        self.global_mean = None
        self.user_features, self.item_features = None, None
        self.user_biases, self.item_biases = None, None
        self._user_id_map, self._item_id_map = None, None
        return

    def fit(self, X: pd.DataFrame):
        """ Decompose user-item rating matrix into thin matrices p and q along with biases

        Arguments:
            X {pandas DataFrame} -- Dataframe containing columns u_id, i_id and rating
        """
        X, self._user_id_map, self._item_id_map = preprocess_data(X)

        self.n_users = len(self._user_id_map)
        self.n_items = len(self._item_id_map)

        # Initialize parameters
        self.user_features = np.random.normal(
            self.init_mean, self.init_sd, (self.n_users, self.n_factors)
        )
        self.item_features = np.random.normal(
            self.init_mean, self.init_sd, (self.n_items, self.n_factors)
        )
        self.user_biases = np.zeros(self.n_users)
        self.item_biases = np.zeros(self.n_items)

        self.global_mean = X["rating"].mean()

        # Perform stochastic gradient descent
        (
            self.user_features,
            self.item_features,
            self.user_biases,
            self.item_biases,
        ) = _sgd(
            X=X.to_numpy(),
            global_mean=self.global_mean,
            user_features=self.user_features,
            item_features=self.item_features,
            user_biases=self.user_biases,
            item_biases=self.item_biases,
            n_epochs=self.n_epochs,
            lr=self.lr,
            reg=self.reg,
            verbose=self.verbose,
        )

        return self

    def predict(self, X: pd.DataFrame):
        """Predict ratings for given users and items

        Arguments:
            X {pd.DataFrame} -- Dataframe containing columns u_id and i_id

        Returns:
            predictions [np.ndarray] -- Vector containing rating predictions of all user, items in same order as input X
            predictions_possible [np.ndarray] -- Vector of whether both given user and item were contained in the data that the model was fitted on
        """
        # Keep only required columns in given order
        X = X.loc[:, ["u_id", "i_id"]]

        # Remap user_id and item_id
        X.loc[:, "u_id"] = X["u_id"].map(self._user_id_map)
        X.loc[:, "i_id"] = X["i_id"].map(self._item_id_map)

        # Replace missing mappings with -1
        X.fillna(-1, inplace = True)

        # Get predictions
        predictions, predictions_possible = _predict(
            X=X.to_numpy(),
            global_mean=self.global_mean,
            min_rating=self.min_rating,
            max_rating=self.max_rating,
            user_features=self.user_features,
            item_features=self.item_features,
            user_biases=self.user_biases,
            item_biases=self.item_biases,
        )

        return (predictions, predictions_possible)
