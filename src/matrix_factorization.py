import numba as nb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from .preprocess import preprocess_data, preprocess_data_predict, preprocess_data_update


# TODO: Train each feature component incrementally
# TODO: Add non-linearity function with sigmoid according to Simon Funk. Kernel functions

# TODO: Add early stopping
# TODO: Save training RMSE values
# TODO: change fit signature to match sklearn
# TODO: Add input checking
# TODO: Abstract update to either matrices and not just user matrix since the MF model is symmetric in terms of user/items
# TODO: Maybe have separate reg and lr param for update?


@nb.njit()
def _rmse(
    X: np.ndarray,
    global_mean: float,
    user_features: np.ndarray,
    item_features: np.ndarray,
    user_biases: np.ndarray,
    item_biases: np.ndarray,
):
    """
    Calculates root mean squared error for given data and model parameters

    Args:
        X (np.ndarray): Matrix with columns user, item and rating
        global_mean (float): Global mean rating
        user_features (np.ndarray): User features matrix P of size (n_users, n_factors)
        item_features (np.ndarray): Item features matrix Q of size (n_items, n_factors)
        user_biases (np.ndarray): User biases vector of shape (n_users, 1)
        item_biases (np.ndarray): Item biases vector of shape (n_items, 1)

    Returns:
        rmse [float]: Root mean squared error
    """
    n_factors = user_features.shape[1]
    n_ratings = X.shape[0]
    errors = np.zeros(n_ratings)

    # Iterate through all user-item ratings
    for i in range(n_ratings):
        user_id, item_id, rating = int(X[i, 0]), int(X[i, 1]), X[i, 2]
        rating_pred = (
            global_mean
            + user_biases[user_id]
            + item_biases[item_id]
            + np.dot(user_features[user_id, :], item_features[item_id, :])
        )
        errors[i] = rating - rating_pred

    rmse = np.sqrt(np.square(errors).mean())

    return rmse


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
    update_user_params: bool = True,
    update_item_params: bool = True,
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Performs stochastic gradient descent. Similar to https://github.com/gbolmier/funk-svd and https://github.com/NicolasHug/Surprise we iterate
    over each factor manually for a given user/item instead of indexing by a row such as user_feature[user] since it has shown to be much faster. 
    We have also tested with representing user_features and item_features as 1D arrays but that also is much slower. Using parallel turned on in numba
    gives much worse performance as well.

    Arguments:
        X {numpy array} -- User-item ranking matrix
        global_mean {float} -- Global mean of all ratings
        user_features {numpy array} -- Start matrix P of user features of shape (n_users, n_factors)
        item_features {numpy array} -- Start matrix Q of item features of shape (n_items, n_factors)
        user_biases {numpy array} -- User biases vector of shape (n_users, 1)
        item_biases {numpy array} -- Item biases vector of shape (n_items, 1)
        n_epochs {int} -- Number of epochs to run
        lr {float} -- Learning rate alpha
        reg {float} -- Regularization parameter lambda for Frobenius norm
        verbose {int} -- Verbosity when fitting. 0 for nothing and 1 for printing epochs
        update_user_params {bool} -- Whether to update user bias parameters or not. Default is True.
        update_item_params {bool} -- Whether to update item bias parameters or not. Default is True.

    Returns:
        user_features [np.ndarray] -- Updated user_features matrix P
        item_features [np.ndarray] -- Updated item_features matrix Q
        user_biases [np.ndarray] -- Updated user_biases vector
        item_biases [np.ndarray] -- Updated item_bases vector
    """
    n_factors = user_features.shape[1]
    error = 0

    for epoch in range(n_epochs):

        # Iterate through all user-item ratings
        for i in range(X.shape[0]):
            user_id, item_id, rating = int(X[i, 0]), int(X[i, 1]), X[i, 2]
            item_bias = item_biases[item_id]
            user_bias = user_biases[user_id]

            # Compute error
            rating_pred = global_mean + item_bias + user_bias
            for f in range(n_factors):
                rating_pred += user_features[user_id, f] * item_features[item_id, f]
            error = rating - rating_pred

            # Update bias parameters
            if update_user_params:
                user_biases[user_id] += lr * (error - reg * user_bias)

            if update_item_params:
                item_biases[item_id] += lr * (error - reg * item_bias)

            # Update P and Q matrices containing latent factor parameters
            for f in range(n_factors):
                user_feature = user_features[user_id, f]
                item_feature = item_features[item_id, f]

                if update_user_params:
                    user_features[user_id, f] += lr * (
                        error * item_feature - reg * user_feature
                    )

                if update_item_params:
                    item_features[item_id, f] += lr * (
                        error * user_feature - reg * item_feature
                    )

        # Calculate error and print
        if verbose == 1:
            rmse = _rmse(
                X=X,
                global_mean=global_mean,
                user_features=user_features,
                item_features=item_features,
                user_biases=user_biases,
                item_biases=item_biases,
            )
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
) -> (list, list):
    """ 
    Calculate predicted ratings for each user-item pair.

    Arguments:
        X {np.ndarray} -- Matrix with columns representing (user_id, item_id)
        global_mean {float} -- Global mean of all ratings
        min_rating {int} -- Lowest rating possible
        max_rating {int} -- Highest rating possible
        user_features {np.ndarray} -- User features matrix P of shape (n_users, n_factors)
        item_features {np.ndarray} -- Item features matrix Q of shape (n_items, n_factors)
        user_biases {np.ndarray} -- User biases vector of length n_users
        item_biases {np.ndarray} -- Item biases vector of length n_items

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

        # If known user or time then add corresponding bias and feature vector product
        if user_known:
            rating_pred += user_biases[user_id]
        if item_known:
            rating_pred += item_biases[item_id]
        if user_known and item_known:
            rating_pred += np.dot(user_features[user_id, :], item_features[item_id, :])

        # Bound ratings to min and max rating range
        if rating_pred > max_rating:
            rating_pred = max_rating
        elif rating_pred < min_rating:
            rating_pred = min_rating

        predictions.append(rating_pred)
        predictions_possible.append(user_known and item_known)

    return predictions, predictions_possible


class MatrixFactorization(BaseEstimator):
    """ 
    Biased Matrix Factorization (FunkSVD) by Simon Funk. Finds the thin matrices P and Q such that P * Q^T give a good low rank approximation to the user-item 
    ratings matrix A based on RMSE. This is different from SVD despite the name as unlike SVD there is no constraint for matrices P and Q to have mutually
    orthogonal columns. This algorithm also only uses the observed user item ratings and does not focus on the priors. 

    Arguments:
        n_factors {int} -- The number of latent factors in matrices P and Q (default: {100})
        n_epochs {int} -- Number of epochs to train for (default: {100})
        reg {float} -- Regularization parameter lambda for Tikhonov regularization (default: {0})
        lr {float} -- Learning rate alpha for gradient optimization step (default: {0.01})
        init_mean {float} -- Mean of normal distribution to use for initializing parameters (default: {0})
        init_sd {float} -- Standard deviation of normal distribution to use for initializing parameters (default: {0.1})
        min_rating {int} -- Smallest rating possible (default: {0})
        max_rating {int} -- Largest rating possible (default: {5})
        verbose {str} -- Verbosity when fitting. 0 to not print anything, 1 to print fitting model (default: {1})

    Attributes:
        n_users {int} -- Number of users
        n_items {int} -- Number of items
        global_mean {float} -- Global mean of all ratings
        user_features {numpy array} -- Decomposed P matrix of user features of shape (n_users, n_factors)
        item_features {numpy array} -- Decomposed Q matrix of item features of shape (n_items, n_factors)
        user_biases {numpy array} -- User bias vector of shape (n_users, 1)
        item_biases {numpy array} -- Item bias vector of shape (n_items, i)
        user_id_map {dict} -- Mapping of user ids to assigned integer ids
        item_id_map {dict} -- Mapping of item ids to assigned integer ids
        _predictions_possible {list} -- Boolean vector of whether both user and item were known for prediction. Only available after calling predict
    """

    def __init__(
        self,
        n_factors: int = 100,
        n_epochs: int = 100,
        reg: float = 0,
        lr: float = 0.01,
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
        self.user_id_map, self.item_id_map = None, None
        return

    def fit(self, X: pd.DataFrame):
        """ 
        Decompose user-item rating matrix into thin matrices P and Q along with biases

        Arguments:
            X {pandas DataFrame} -- Dataframe containing columns user_id, item_id and rating
        """
        X, self.user_id_map, self.item_id_map = preprocess_data(X)

        self.n_users = len(self.user_id_map)
        self.n_items = len(self.item_id_map)

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

    def predict(self, X: pd.DataFrame) -> list:
        """
        Predict ratings for given users and items

        Arguments:
            X {pd.DataFrame} -- Dataframe containing columns user_id and item_id

        Returns:
            predictions [np.ndarray] -- Vector containing rating predictions of all user, items in same order as input X
        """
        # If empty return empty list
        if X.shape[0] == 0:
            return []

        X = preprocess_data_predict(
            X=X, user_id_map=self.user_id_map, item_id_map=self.item_id_map
        )

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

        self._predictions_possible = predictions_possible

        return predictions

    def update_users(
        self, X: pd.DataFrame, lr: float = 0.01, n_epochs: int = 20, verbose: int = 0
    ):
        """
        Update P user features matrix with new/updated user-item ratings information using SGD. Only the user parameters corresponding for the
        new/updated users will be updated and item parameters will be left alone.

        Note: If updating old users then pass all user-item ratings for old users and not just modified ratings

        Args:
            X (pd.DataFrame): Dataframe containing columns user_id, item_id and rating
            lr (float, optional): Learning rate alpha for gradient optimization step
            n_epochs (int, optional): Number of epochs to run SGD. Defaults to 20.
            verbose (int, optional): Verbosity when updating, 0 for nothing and 1 for training messages. Defaults to 0.
        """
        X, self.user_id_map, old_users, new_users = preprocess_data_update(
            X=X, user_id_map=self.user_id_map, item_id_map=self.item_id_map
        )

        n_new_users = len(new_users)

        # Re-initialize params for old users
        for user in old_users:
            user_index = self.user_id_map[user]

            # Initialize latent factors vector
            self.user_features[user_index, :] = np.random.normal(
                self.init_mean, self.init_sd, (1, self.n_factors)
            )

            # Initialize bias
            self.user_biases[user_index] = 0

        # Add bias parameters for new users
        self.user_biases = np.append(self.user_biases, np.zeros(n_new_users))

        # Add latent factor parameters for new users by adding rows to P matrix
        new_user_features = np.random.normal(
            self.init_mean, self.init_sd, (n_new_users, self.n_factors)
        )
        self.user_features = np.concatenate(
            (self.user_features, new_user_features), axis=0
        )

        # Estimate new parameters
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
            n_epochs=n_epochs,
            lr=lr,
            reg=self.reg,
            verbose=verbose,
            update_item_params=False,
        )

        return
